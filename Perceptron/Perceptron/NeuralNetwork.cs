using System;

public class NeuralNetwork
{
    private static Random random = new Random();
    public double[,] WeightsInputHidden { get; private set; }
    public double[,] WeightsHiddenOutput { get; private set; }
    public double[] HiddenLayerOutput { get; private set; }
    public double[] OutputLayerOutput { get; private set; }
    public int InputNodes { get; private set; }
    public int HiddenNodes { get; private set; }
    public int OutputNodes { get; private set; }
    public double LearningRate { get; set; }

    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
    {
        InputNodes = inputNodes;
        HiddenNodes = hiddenNodes;
        OutputNodes = outputNodes;
        LearningRate = learningRate;

        WeightsInputHidden = new double[inputNodes, hiddenNodes];
        InitializeWeights(WeightsInputHidden);

        WeightsHiddenOutput = new double[hiddenNodes, outputNodes];
        InitializeWeights(WeightsHiddenOutput);

        HiddenLayerOutput = new double[hiddenNodes];
        OutputLayerOutput = new double[outputNodes];
    }

    private void InitializeWeights(double[,] weights)
    {
        int fromNodes = weights.GetLength(0);
        int toNodes = weights.GetLength(1);
        double limit = Math.Sqrt(6 / (fromNodes + toNodes));

        for (int i = 0; i < fromNodes; i++)
            for (int j = 0; j < toNodes; j++)
                weights[i, j] = random.NextDouble() * (2 * limit) - limit;
    }

    public void Train(double[] inputs, double[] targets)
    {
        FeedForward(inputs);

        double[] outputErrors = new double[OutputNodes];
        for (int i = 0; i < OutputNodes; i++)
            outputErrors[i] = targets[i] - OutputLayerOutput[i];

        double[] hiddenErrors = new double[HiddenNodes];
        for (int i = 0; i < HiddenNodes; i++)
            for (int j = 0; j < OutputNodes; j++)
                hiddenErrors[i] += WeightsHiddenOutput[i, j] * outputErrors[j];

        for (int i = 0; i < HiddenNodes; i++)
            for (int j = 0; j < OutputNodes; j++)
                WeightsHiddenOutput[i, j] += LearningRate * outputErrors[j] * HiddenLayerOutput[i];

        for (int i = 0; i < InputNodes; i++)
            for (int j = 0; j < HiddenNodes; j++)
                WeightsInputHidden[i, j] += LearningRate * hiddenErrors[j] * inputs[i];
    }

    public double[] FeedForward(double[] inputs)
    {
        for (int i = 0; i < HiddenNodes; i++)
        {
            double sum = 0;
            for (int j = 0; j < InputNodes; j++)
                sum += WeightsInputHidden[j, i] * inputs[j];
            HiddenLayerOutput[i] = Sigmoid(sum);
        }

        for (int i = 0; i < OutputNodes; i++)
        {
            double sum = 0;
            for (int j = 0; j < HiddenNodes; j++)
                sum += WeightsHiddenOutput[j, i] * HiddenLayerOutput[j];
            OutputLayerOutput[i] = Sigmoid(sum);
        }

        return OutputLayerOutput;
    }

    private double Sigmoid(double value)
    {
        return 1.0 / (1.0 + Math.Exp(-value));
    }
}
