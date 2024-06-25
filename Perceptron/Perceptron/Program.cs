using System;
using System.Globalization;
using System.IO;

class Program
{
    static void Main()
    {
        var trainingData = LoadData("C:/Users/early/Desktop/Perceptron/Perceptron/training_data.txt");
        int inputNodes = 5;
        int hiddenNodes = 10;
        int outputNodes = 3;
        double learningRate = 0.1;
        NeuralNetwork network = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);

        int epochs = 1000;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var data in trainingData)
            {
                network.Train(data.Item1, data.Item2);
            }
        }

        foreach (var data in trainingData)
        {
            var outputs = network.FeedForward(data.Item1);
            Console.WriteLine($"Expected: {string.Join(", ", data.Item2)}, Got: {string.Join(", ", outputs)}");
        }
    }

    static Tuple<double[], double[]>[] LoadData(string filename)
    {
        string[] lines = File.ReadAllLines(filename);
        var culture = CultureInfo.InvariantCulture;
        var data = new Tuple<double[], double[]>[lines.Length];

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Split(',');
            var inputs = new double[5];
            var outputs = new double[3];

            for (int j = 0; j < 5; j++)
                inputs[j] = double.Parse(line[j], culture);

            for (int j = 5; j < 8; j++)
                outputs[j - 5] = double.Parse(line[j], culture);

            data[i] = new Tuple<double[], double[]>(inputs, outputs);
        }

        return data;
    }
}
