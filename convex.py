import numpy as np
import matplotlib.pyplot as plt



N_UE = 100                 # Number of User Equipments
ROUNDS = 40                # Communication rounds
LOCAL_EPOCHS = 2           # Local training epochs
LEARNING_RATE = 0.05
L2_REG = 1e-4


RADIUS = 50.0              # meters
ALPHA = 2.0                # path loss factor
PM = 1.0                   # mW
P0_DBM = -65.0             # dBm

# Total communication trial budget
TOTAL_TRIALS = 400

# Dataset
TRAIN_SIZE = 12000
TEST_SIZE = 3000

RANDOM_SEED = 42




def dbm_to_mw(dbm):
    return 10 ** (dbm / 10.0)


P0 = dbm_to_mw(P0_DBM)




def sigmoid(z):

    # Prevent overflow in exp()
    z = np.clip(z, -50, 50)

    return 1.0 / (1.0 + np.exp(-z))



def logistic_loss(w, X, y):

    probabilities = sigmoid(X @ w)

    eps = 1e-12

    loss = -np.mean(
        y * np.log(probabilities + eps)
        +
        (1 - y) * np.log(1 - probabilities + eps)
    )

    # L2 regularization
    
    # Do not regularize bias.
    loss += 0.5 * L2_REG * np.sum(w[:-1] ** 2)

    return loss


def logistic_gradient(w, X, y):

    probabilities = sigmoid(X @ w)

    gradient = (X.T @ (probabilities - y)) / len(X)

    # L2 regularization
    gradient[:-1] += L2_REG * w[:-1]

    return gradient



def local_train(global_weights, X, y):

    local_weights = global_weights.copy()

    for _ in range(LOCAL_EPOCHS):

        grad = logistic_gradient(
            local_weights,
            X,
            y
        )

        local_weights -= LEARNING_RATE * grad

    return local_weights




def calculate_accuracy(w, X, y):

    probabilities = sigmoid(X @ w)

    predictions = (probabilities >= 0.5).astype(int)

    return np.mean(predictions == y)



def create_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    total = TRAIN_SIZE + TEST_SIZE

    # Number of features
    D = 30

    # Generate features
    X = rng.normal(
        0,
        1,
        size=(total, D)
    )

    # True model
    true_w = rng.normal(
        0,
        1,
        size=D
    )

    # Generate probabilities
    logits = X @ true_w

    probabilities = sigmoid(logits)

    # Generate labels
    y = (
        rng.random(total)
        <
        probabilities
    ).astype(int)

    # Shuffle
    indices = rng.permutation(total)

    X = X[indices]
    y = y[indices]

    X_train = X[:TRAIN_SIZE]
    y_train = y[:TRAIN_SIZE]

    X_test = X[TRAIN_SIZE:]
    y_test = y[TRAIN_SIZE:]

    # Add bias feature
    X_train = np.c_[
        X_train,
        np.ones(len(X_train))
    ]

    X_test = np.c_[
        X_test,
        np.ones(len(X_test))
    ]

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )




def split_dataset(X, y):

    rng = np.random.default_rng(RANDOM_SEED)

    indices = np.arange(len(X))

    rng.shuffle(indices)

    chunks = np.array_split(
        indices,
        N_UE
    )

    clients = []

    for chunk in chunks:

        X_local = X[chunk]
        y_local = y[chunk]

        clients.append(
            (
                X_local,
                y_local
            )
        )

    return clients




def generate_ue_locations():

    rng = np.random.default_rng(RANDOM_SEED)

    r = (
        RADIUS
        *
        np.sqrt(rng.random(N_UE))
    )

    theta = (
        2
        *
        np.pi
        *
        rng.random(N_UE)
    )

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    distances = np.sqrt(
        x ** 2 +
        y ** 2
    )

    return distances




def kkt_continuous_allocation(
    probabilities,
    total_trials
):

    probabilities = np.clip(
        probabilities,
        1e-12,
        1 - 1e-12
    )

    # bi = -log(1-pi)
    b = -np.log(
        1 - probabilities
    )

    def total_trials_for_mu(mu):

        t = (
            np.log(
                1 + b / mu
            )
            /
            b
        )

        return np.sum(t)


    low = 1e-12
    high = 1.0

    while (
        total_trials_for_mu(high)
        >
        total_trials
    ):

        high *= 2


    for _ in range(200):

        mid = (
            low +
            high
        ) / 2

        current = total_trials_for_mu(mid)

        if current > total_trials:

            # Need larger mu
            low = mid

        else:

            # Need smaller mu
            high = mid

    mu0 = (
        low +
        high
    ) / 2

    # Calculate continuous t_i

    t = (
        np.log(
            1 + b / mu0
        )
        /
        b
    )

    return t, mu0





def integerize_trials(
    continuous_trials,
    total_trials
):

    n = len(
        continuous_trials
    )

    # First floor
    trials = np.floor(
        continuous_trials
    ).astype(int)

    # Every UE gets at least one trial
    trials = np.maximum(
        trials,
        1
    )

   

    while np.sum(trials) > total_trials:

        candidates = np.where(
            trials > 1
        )[0]

        if len(candidates) == 0:
            break

        fractional = (
            continuous_trials
            -
            np.floor(
                continuous_trials
            )
        )

        # Remove from smallest fractional part
        index = candidates[
            np.argmin(
                fractional[candidates]
            )
        ]

        trials[index] -= 1

   

    remaining = (
        total_trials
        -
        np.sum(trials)
    )

    fractional = (
        continuous_trials
        -
        np.floor(
            continuous_trials
        )
    )

    order = np.argsort(
        -fractional
    )

    for i in range(
        remaining
    ):

        trials[
            order[
                i % n
            ]
        ] += 1

    return trials




def equal_trial_allocation():

    trials = np.full(
        N_UE,
        TOTAL_TRIALS // N_UE,
        dtype=int
    )

    remainder = (
        TOTAL_TRIALS
        %
        N_UE
    )

    for i in range(remainder):

        trials[i] += 1

    return trials




def transmission_success(
    p,
    trials,
    rng
):

    if trials <= 0:

        return False

    success_probability = (
        1
        -
        (1 - p) ** trials
    )

    random_number = rng.random()

    return (
        random_number
        <
        success_probability
    )




def aggregate_models(
    local_models,
    successful_clients,
    clients
):

    if len(successful_clients) == 0:

        return None

    total_data = 0

    for i in successful_clients:

        total_data += len(
            clients[i][0]
        )

    # Start with zero vector
    global_model = np.zeros_like(
        local_models[0]
    )

    for i in successful_clients:

        weight = (
            len(clients[i][0])
            /
            total_data
        )

        global_model += (
            weight
            *
            local_models[i]
        )

    return global_model



def run_experiment(
    name,
    clients,
    probabilities,
    trial_allocations,
    X_test,
    y_test,
    seed
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    rng = np.random.default_rng(seed)

    # Number of model parameters
    dimension = X_test.shape[1]

    # Initial global model
    global_weights = np.zeros(
        dimension
    )

    accuracy_history = []

    participation_history = []

    loss_history = []

    successful_ue_history = []



    for round_number in range(
        ROUNDS
    ):

        local_models = []

        successful_clients = []

      

        for i in range(N_UE):

            X_local, y_local = clients[i]

            local_weights = local_train(
                global_weights,
                X_local,
                y_local
            )

            local_models.append(
                local_weights
            )

        

            success = transmission_success(
                probabilities[i],
                trial_allocations[i],
                rng
            )

            if success:

                successful_clients.append(i)

       

        aggregated_model = aggregate_models(
            local_models,
            successful_clients,
            clients
        )

    
        if aggregated_model is not None:

            global_weights = (
                aggregated_model
            )


        accuracy = calculate_accuracy(
            global_weights,
            X_test,
            y_test
        )

        loss = logistic_loss(
            global_weights,
            X_test,
            y_test
        )

        participation = (
            len(successful_clients)
            /
            N_UE
        )

        # Store
        accuracy_history.append(
            accuracy
        )

        loss_history.append(
            loss
        )

        participation_history.append(
            participation
        )

        successful_ue_history.append(
            len(successful_clients)
        )

        print(
            f"Round {round_number + 1:02d} | "
            f"Accuracy = {accuracy:.4f} | "
            f"Loss = {loss:.4f} | "
            f"Successful UEs = "
            f"{len(successful_clients)}/{N_UE}"
        )

    return {
        "accuracy": accuracy_history,
        "loss": loss_history,
        "participation": participation_history,
        "successful": successful_ue_history
    }



def main():

    print()
    print("=" * 70)
    print("DISTRIBUTED LEARNING FOR CONVEX LOGISTIC REGRESSION")
    print("=" * 70)

    print()
    print("Number of UEs       :", N_UE)
    print("Communication rounds:", ROUNDS)
    print("Total trial budget  :", TOTAL_TRIALS)
    print("Radius              :", RADIUS, "m")
    print("Path loss alpha     :", ALPHA)
    print("Transmit power      :", PM, "mW")
    print("Minimum received P  :", P0_DBM, "dBm")


    print()
    print("Creating dataset...")

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = create_dataset()

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples :",
        len(X_test)
    )

 

    print()
    print("Distributing dataset across UEs...")

    clients = split_dataset(
        X_train,
        y_train
    )

 
    print()
    print("Generating UE locations...")

    distances = generate_ue_locations()

   

    probabilities = (
        calculate_participation_probabilities(
            distances
        )
    )

    print()
    print("First 10 UE distances:")

    print(
        np.round(
            distances[:10],
            3
        )
    )

    print()
    print("First 10 participation probabilities:")

    print(
        np.round(
            probabilities[:10],
            5
        )
    )

 

    print()
    print("=" * 70)
    print("KKT TRIAL ALLOCATION")
    print("=" * 70)

    continuous_trials, mu0 = (
        kkt_continuous_allocation(
            probabilities,
            TOTAL_TRIALS
        )
    )

    print()
    print("mu_0 =", mu0)

    print()
    print("Continuous trial allocation:")

    print(
        np.round(
            continuous_trials,
            3
        )
    )


    optimized_trials = (
        integerize_trials(
            continuous_trials,
            TOTAL_TRIALS
        )
    )

    print()
    print("Integer KKT allocation:")

    print(
        optimized_trials
    )

    print()
    print(
        "Total allocated:",
        np.sum(
            optimized_trials
        )
    )



    equal_trials = (
        equal_trial_allocation()
    )

    print()
    print("Equal allocation:")

    print(
        equal_trials
    )

    print()
    print(
        "Total allocated:",
        np.sum(equal_trials)
    )

 
    optimized_history = (
        run_experiment(
            "KKT / OPTIMIZED TRIAL ALLOCATION",
            clients,
            probabilities,
            optimized_trials,
            X_test,
            y_test,
            100
        )
    )


    equal_history = (
        run_experiment(
            "EQUAL TRIAL ALLOCATION",
            clients,
            probabilities,
            equal_trials,
            X_test,
            y_test,
            200
        )
    )

 

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()

    print(
        "Final KKT accuracy:",
        optimized_history["accuracy"][-1]
    )

    print(
        "Final Equal accuracy:",
        equal_history["accuracy"][-1]
    )

    print()

    print(
        "Average KKT participation:",
        np.mean(
            optimized_history[
                "participation"
            ]
        )
    )

    print(
        "Average Equal participation:",
        np.mean(
            equal_history[
                "participation"
            ]
        )
    )



    rounds = np.arange(
        1,
        ROUNDS + 1
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        rounds,
        optimized_history["accuracy"],
        label="KKT allocation"
    )

    plt.plot(
        rounds,
        equal_history["accuracy"],
        label="Equal allocation"
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Test Accuracy"
    )

    plt.title(
        "Accuracy vs Communication Rounds"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "accuracy_vs_rounds.png",
        dpi=150
    )


    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        rounds,
        optimized_history[
            "participation"
        ],
        label="KKT allocation"
    )

    plt.plot(
        rounds,
        equal_history[
            "participation"
        ],
        label="Equal allocation"
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Successful UE Fraction"
    )

    plt.title(
        "UE Participation vs Communication Rounds"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "participation_vs_rounds.png",
        dpi=150
    )

  

    plt.show()



if __name__ == "__main__":

    main()