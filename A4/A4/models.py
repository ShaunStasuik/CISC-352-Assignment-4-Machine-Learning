import nn

class PerceptronModel(object):
    def __init__(self, dimension):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimension` is the dimensionality of the data.
        For example, dimension=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimension)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, point):
        """
        Calculates the score assigned by the perceptron to a data point `point`.

        Inputs:
            point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, point)

    def get_prediction(self, point):
        """
        Calculates the predicted class for a single data point `point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(self.run(point))
        if score >= 0:
            return 1
        else:
            return -1

    def train_model(self, data):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            made_mistake = False

            for x, y in data.iterate_once(1):
                true_label = nn.as_scalar(y)
                prediction = self.get_prediction(x)

                if prediction != true_label:
                    self.w.update(true_label, x)
                    made_mistake = True

            if not made_mistake:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # initializing model parameters
        # layer 1 - input 1 feature -> 512 hidden units
        self.w1 = nn.Parameter(1, 512)
        self.b1 = nn.Parameter(1, 512)

        # layer 2 - 512 -> 512 hidden units
        self.w2 = nn.Parameter(512, 512)
        self.b2 = nn.Parameter(1, 512)

        # layer 3 - 512 -> output 1 val
        self.w3 = nn.Parameter(512, 1)
        self.b3 = nn.Parameter(1, 1)

        # learning rate
        self.lr = 0.005

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # layer 1 - linear transformation, bias, ReLU activation
        layer1 = nn.ReLU(nn.AddBias(self.b1, nn.Linear(x, self.w1)))

        # layer 2 - linear transformation, bias, ReLU activation
        layer2 = nn.ReLU(nn.AddBias(self.b2, nn.Linear(layer1, self.w2)))

        # output - linear transformation, bias 
        output = nn.AddBias(self.b3, nn.Linear(layer2, self.w3))

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)


    def train_model(self, data):
        """
        Trains the model.
        """
        while True:
            total_loss = 0
            num_batches = 0

            # iterating over mini batches of dataset
            for x, y in data.iterate_once(batch_size=200):
                loss = self.get_loss(x, y)

                # collecting all parameter
                params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

                # computing gradients
                grads = nn.gradients(params, loss)

                # updating each parameter 
                for param, grad in zip(params, grads):
                    param.update(-self.lr, grad)

                total_loss += nn.as_scalar(loss)
                num_batches += 1

            # computing average loss 
            avg_loss = total_loss / num_batches

            # stopping training once average loss is less than 0.02
            if avg_loss <= 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # Hyperparameters
        self.batch_size = 100
        self.learning_rate = 0.1

        # 784 -> 200 -> 100 -> 10
        self.w1 = nn.Parameter(784, 200)
        self.b1 = nn.Parameter(1, 200)

        self.w2 = nn.Parameter(200, 100)
        self.b2 = nn.Parameter(1, 100)

        self.w3 = nn.Parameter(100, 10)
        self.b3 = nn.Parameter(1, 10)

        self.parameters = [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3
        ]
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        h1 = nn.ReLU(nn.AddBias(self.b1, nn.Linear(x, self.w1)))
        h2 = nn.ReLU(nn.AddBias(self.b2, nn.Linear(h1, self.w2)))
        logits = nn.AddBias(self.b3, nn.Linear(h2, self.w3))
        return logits

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train_model(self, data):
        """
        Trains the model.
        """
        target_validation_accuracy = 0.975
        max_epochs = 20

        for _ in range(max_epochs):
            for x, y in data.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(self.parameters, loss)

                for parameter, gradient in zip(self.parameters, gradients):
                    parameter.update(-self.learning_rate, gradient)

            if data.get_validation_accuracy() >= target_validation_accuracy:
                break

