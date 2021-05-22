﻿using MachineLearning.Calculus;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    public class SingleLayerPerceptron : IMachineLearning
    {
		#region Private Contants
		const double LEARNING_RATE = 0.1;
		#endregion

		#region Public Method
		public int NumberInput;
        public int NumberNeurons;
        public int NumberOutput;
		#endregion

		#region Private Properties
		private List<List<double>> weightsIH;
        private List<List<double>> biasIH;
        private List<List<double>> weightsHO;
        private List<List<double>> biasHO;
        private List<List<double>> ActivationHiddenSigmoid;
        #endregion

        #region Ctor
        public SingleLayerPerceptron(int numberInput, int nNeurons, int numberOutput)
        {
            NumberInput = numberInput;
            NumberNeurons = nNeurons;
            NumberOutput = numberOutput;

            weightsIH = new List<List<double>>();
            weightsHO = new List<List<double>>();
            biasIH = new List<List<double>>();
            biasHO = new List<List<double>>();

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
        #endregion

        #region Public Methods
        public List<double> Predict(List<double> datas)
        {
            List<List<double>> matrix = MatrixCalculus.GetMatrix(datas);
            List<List<double>> result = Predict(matrix);
            List<double> output = MatrixCalculus.ToArray(result);

            return output;
        }

        public List<List<double>> Predict(List<List<double>> data)
        {
            List<List<double>> activationHidden = MatrixCalculus.Add(MatrixCalculus.Multiply(weightsIH, data), biasIH);

            ActivationHiddenSigmoid = MatrixCalculus.Sigmoid(activationHidden);

            List<List<double>> output = MatrixCalculus.Add(MatrixCalculus.Multiply(weightsHO, ActivationHiddenSigmoid), biasHO);
            List<List<double>> activationOutputSigmoid = MatrixCalculus.Sigmoid(output);

            return activationOutputSigmoid;
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
            List<List<double>> resultMatrix = MatrixCalculus.GetMatrix(expectedValues);
            List<List<double>> inputMatrix = MatrixCalculus.GetMatrix(datas);

            List<List<double>> guess = Predict(inputMatrix);

            //Hidden-Out
            List<List<double>> errorMatrix = MatrixCalculus.Substract(resultMatrix, guess);
            List<List<double>> deltaWeightsHO = MatrixCalculus.Multiply(GetDeltaError(errorMatrix, guess), MatrixCalculus.Transpose(ActivationHiddenSigmoid));
            List<List<double>> deltaBiasHO = GetDeltaError(errorMatrix, guess);

            //In-Hidden
            List<List<double>> errorHiddenMatrix = MatrixCalculus.Multiply(MatrixCalculus.Transpose(weightsHO), errorMatrix);
            List<List<double>> deltaWeightsIH = MatrixCalculus.Multiply(GetDeltaError(errorHiddenMatrix, ActivationHiddenSigmoid), MatrixCalculus.Transpose(inputMatrix));
            List<List<double>> deltaBiasIH = GetDeltaError(errorHiddenMatrix, ActivationHiddenSigmoid);

            weightsHO = MatrixCalculus.Add(weightsHO, deltaWeightsHO);
            weightsIH = MatrixCalculus.Add(weightsIH, deltaWeightsIH);

            biasHO = MatrixCalculus.Add(biasHO, deltaBiasHO);
            biasIH = MatrixCalculus.Add(biasIH, deltaBiasIH);
        }
		#endregion

		#region Private Methods
		private List<List<double>> GetDeltaError(List<List<double>> errorMatrix, List<List<double>> resultMatrix)
        {
            int n = resultMatrix.Count;
            int m = resultMatrix[0].Count;

            List<List<double>> dSigmoid = MatrixCalculus.Map(resultMatrix, derivativeSigmoid);

            return MatrixCalculus.Multiply(
                MatrixCalculus.HadamarProduct(
                    errorMatrix
                , dSigmoid)
                , LEARNING_RATE);
        }

        private static double derivativeSigmoid(double number)
        {
            return number * (1 - number);
        }
		#endregion
    }
}