using MachineLearning.Calculus;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    public class DeepLearning
    {
        public int NumberInput;
        public int NumberNeurons;
        public int NumberOutput;
        public int NumberOfLayers;

        const double m_LearningRate = 0.1;

        //List<List<double>> weightsIH;
        //List<List<double>> biasIH;

        //List<List<double>> weightsHO;
        //List<List<double>> biasHO;

        /// <summary>
        /// List Of Matrix
        /// </summary>
        List<List<List<double>>> weightsHidden;

        /// <summary>
        /// List Of Matrix
        /// </summary>
        List<List<List<double>>> biasHidden;

        /// <summary>
        /// List of all the Activations Matrix of the hidden layers
        /// </summary>
        List<List<List<double>>> ActivationsHiddenSigmoid;
        List<List<List<double>>> Errors;

        public DeepLearning(int numberInput, int nNeurons, int numberOutput, int numberOfLayers)
        {
            NumberInput = numberInput;
            NumberNeurons = nNeurons;
            NumberOutput = numberOutput;
            NumberOfLayers = numberOfLayers;

            Random random = new Random(Guid.NewGuid().GetHashCode());

            //weightsIH = new List<List<double>>();
            weightsHidden = new List<List<List<double>>>();
            //biasIH = new List<List<double>>();
            biasHidden = new List<List<List<double>>>();
            ActivationsHiddenSigmoid = new List<List<List<double>>>();
            Errors = new List<List<List<double>>>();

            //The first layer is different because it is connected to all the Inputs
            List<List<double>> weightsIH = new List<List<double>>();
            List<List<double>> biasIH = new List<List<double>>();

            weightsHidden.Add(weightsIH);
            biasHidden.Add(biasIH);

            //Set the first hidden layer with radom values
            for (int i = 0; i < NumberNeurons; i++)
            {
                List<double> row = new List<double>();
                weightsIH.Add(row);

                for (int j = 0; j < NumberInput; j++)
                    row.Add((new Random(Guid.NewGuid().GetHashCode()).NextDouble()) / 10);

                List<double> rowB = new List<double>();
                rowB.Add((new Random(Guid.NewGuid().GetHashCode()).NextDouble()) / 10);
                biasIH.Add(rowB);
            }

            //Set the hidden weights with random values
            for (int k = 0; k < NumberOfLayers - 1; k++)
            {
                List<List<double>> weights = new List<List<double>>();
                List<List<double>> bias = new List<List<double>>();
                weightsHidden.Add(weights);
                biasHidden.Add(bias);

                for (int i = 0; i < NumberNeurons; i++)
                {
                    List<double> row = new List<double>();
                    weights.Add(row);

                    for (int j = 0; j < NumberNeurons; j++)
                        row.Add((new Random(Guid.NewGuid().GetHashCode()).NextDouble()) / 10);

                    List<double> rowB = new List<double>();
                    rowB.Add((new Random(Guid.NewGuid().GetHashCode()).NextDouble()) / 10);
                    bias.Add(rowB);
                }
            }

            List<List<double>> biasHO = new List<List<double>>();
            List<List<double>> weightsHO = new List<List<double>>();
            weightsHidden.Add(weightsHO);
            biasHidden.Add(biasHO);

            //Set the last layer of neurons with radoms values
            for (int i = 0; i < NumberOutput; i++)
            {
                List<double> row = new List<double>();
                weightsHO.Add(row);

                for (int j = 0; j < NumberNeurons; j++)
                    row.Add((new Random(Guid.NewGuid().GetHashCode()).NextDouble()) / 10);

                List<double> rowB = new List<double>();
                rowB.Add((new Random(Guid.NewGuid().GetHashCode()).NextDouble()) / 10);
                biasHO.Add(rowB);
            }
        }


        public List<double> Predict(List<double> datas)
        {
            List<List<double>> matrix = MatrixCalculus.GetMatrix(datas);
            List<List<double>> result = Predict(matrix);
            List<double> output = MatrixCalculus.ToArray(result);

            return output;
        }

        /// <summary>
        /// Seems to Work
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public List<List<double>> Predict(List<List<double>> data)
        {
            ActivationsHiddenSigmoid.Clear();

            List<List<double>> activationHidden = new List<List<double>>();
            activationHidden = MatrixCalculus.Add(MatrixCalculus.Multiply(weightsHidden[0], data), biasHidden[0]);
            ActivationsHiddenSigmoid.Add(MatrixCalculus.Sigmoid(activationHidden));

            for (int i = 1; i < NumberOfLayers + 1; i++)
            {
                List<List<double>> act = MatrixCalculus.Add(MatrixCalculus.Multiply(weightsHidden[i], ActivationsHiddenSigmoid[i - 1]), biasHidden[i]);
                List<List<double>> sig = MatrixCalculus.Sigmoid(act);
                ActivationsHiddenSigmoid.Add(sig);
            }

            return ActivationsHiddenSigmoid[ActivationsHiddenSigmoid.Count - 1];
        }

        public void Train(List<List<double>> dataSet, List<List<double>> expectedValues)
        {
            for (int i = 0; i < dataSet.Count; i++)
            {
                Train(dataSet[i], expectedValues[i]);
            }
        }

        public void Train(List<double> datas, List<double> expectedValues)
        {
            Errors.Clear();
            List<List<List<double>>> deltaWeights = new List<List<List<double>>>();
            List<List<List<double>>> deltaBias = new List<List<List<double>>>();

            List<List<double>> resultMatrix = GetMatrix(expectedValues);
            List<List<double>> inputMatrix = GetMatrix(datas);

            List<List<double>> guess = Predict(inputMatrix);
            List<List<double>> errorMatrix = MatrixCalculus.Substract(resultMatrix, guess);
            Errors.Add(errorMatrix);

            List<List<double>> deltaWeightsHO = MatrixCalculus.Multiply(GetDeltaError(errorMatrix, guess), MatrixCalculus.Transpose(ActivationsHiddenSigmoid[NumberOfLayers - 1]));
            deltaWeights.Add(deltaWeightsHO);
            List<List<double>> deltaBiasHO = GetDeltaError(errorMatrix, guess);
            deltaBias.Add(deltaBiasHO);

            for (int i = 1; i < NumberOfLayers; i++)
            {
                //weightsHidden is backpropagated but Errors and deltaWeights not !!!
                List<List<double>> errorHiddenMatrix = MatrixCalculus.Multiply(MatrixCalculus.Transpose(weightsHidden[NumberOfLayers + 1 - i]), Errors[i - 1]);
                Errors.Add(errorHiddenMatrix);
                List<List<double>> input_Hidden_T = MatrixCalculus.Transpose(ActivationsHiddenSigmoid[NumberOfLayers - i - 1]);
                List<List<double>> output_Hidden_T = ActivationsHiddenSigmoid[NumberOfLayers - i];
                List<List<double>> gradient = GetDeltaError(errorHiddenMatrix, output_Hidden_T);

                List<List<double>> deltaWeightsHH = MatrixCalculus.Multiply(gradient, input_Hidden_T);
                deltaWeights.Add(deltaWeightsHH);
                List<List<double>> deltaBiasHH = gradient;
                deltaBias.Add(deltaBiasHH);
            }

            List<List<double>> errorFirstMatrix = MatrixCalculus.Multiply(MatrixCalculus.Transpose(weightsHidden[1]), Errors[NumberOfLayers - 1]);
            List<List<double>> output_First_T = ActivationsHiddenSigmoid[0];
            List<List<double>> gradient2 = GetDeltaError(errorFirstMatrix, output_First_T);

            List<List<double>> deltaWeightsIH = MatrixCalculus.Multiply(gradient2, MatrixCalculus.Transpose(inputMatrix));
            deltaWeights.Add(deltaWeightsIH);
            List<List<double>> deltaBiasIH = gradient2;
            deltaBias.Add(deltaBiasIH);

            for (int i = 0; i < deltaWeights.Count; i++)
            {
                weightsHidden[i] = MatrixCalculus.Add(weightsHidden[i], deltaWeights[deltaWeights.Count - 1 - i]);
            }

            //weightsHO = MatrixCalculus.Add(weightsHO, deltaWeightsHO);
            //weightsIH = MatrixCalculus.Add(weightsIH, deltaWeightsIH);

            //biasHO = MatrixCalculus.Add(biasHO, deltaBiasHO);
            //biasIH = MatrixCalculus.Add(biasIH, deltaBiasIH);
        }

        private List<List<double>> GetDeltaError(List<List<double>> errorMatrix, List<List<double>> resultMatrix)
        {
            int n = resultMatrix.Count;
            int m = resultMatrix[0].Count;

            List<List<double>> dSigmoid = MatrixCalculus.Map(resultMatrix, derivativeSigmoid);

            return MatrixCalculus.Multiply(
                MatrixCalculus.HadamarProduct(
                    errorMatrix
                , dSigmoid)
                , m_LearningRate);
        }

        private static List<List<double>> GetMatrix(List<double> expectedValues)
        {
            List<List<double>> expectedValues2 = new List<List<double>>();
            foreach (double value in expectedValues)
                expectedValues2.Add(new List<double>() { value });

            return expectedValues2;
        }

        private static double derivativeSigmoid(double number)
        {
            return number * (1 - number);
        }
    }
}