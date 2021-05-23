using MachineLearning.Calculus;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace MachineLearning
{
	public class SingleLayerPerceptron //: IMachineLearning
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
		private Matrix<double> Wih;
		private Matrix<double> Bih;
		private Matrix<double> Who;
		private Matrix<double> Bho;
		#endregion

		#region Ctor
		public SingleLayerPerceptron(int numberInput, int nNeurons, int numberOutput)
		{
			NumberInput = numberInput;
			NumberNeurons = nNeurons;
			NumberOutput = numberOutput;
			Wih = CreateMatrix.Random<double>(NumberNeurons, NumberInput, new MathNet.Numerics.Distributions.ContinuousUniform(0, 0.1));
			Bih = CreateMatrix.Random<double>(NumberNeurons, 1, new MathNet.Numerics.Distributions.ContinuousUniform(0, 0.1));

			Who = CreateMatrix.Random<double>(NumberOutput, NumberNeurons, new MathNet.Numerics.Distributions.ContinuousUniform(0, 0.1));
			Bho = CreateMatrix.Random<double>(NumberOutput, 1, new MathNet.Numerics.Distributions.ContinuousUniform(0, 0.1));

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
		public double[] Predict(List<double> datas)
		{
			List<List<double>> matrix = MatrixCalculus.GetMatrix(datas);
			var Datas = CreateMatrix.DenseOfRows<double>(matrix);
			var result = Predict(matrix);
			//List<double> output = MatrixCalculus.ToArray(result);
			var array = result.ToArray();
			double[] ar = new double[array.Length];
			int i = 0;
			foreach (var arr in array)
			{
				ar[i++] = arr;
			}
			return ar;
			//return output;
		}

		Matrix<double> activationHidden2;
		private Matrix<double> Predict(List<List<double>> data)
		{
			var Datas = CreateMatrix.DenseOfRows<double>(data);
			activationHidden2 = Wih.Multiply(Datas) + Bih;
			List<List<double>> activationHidden = MatrixCalculus.Add(MatrixCalculus.Multiply(weightsIH, data), biasIH);

			activationHidden2 = activationHidden2.Map(MatrixCalculus.Sigmoid);
			ActivationHiddenSigmoid = MatrixCalculus.Sigmoid(activationHidden);

			var o = Who.Multiply(activationHidden2) + Bho;
			o = o.Map(MatrixCalculus.Sigmoid);
			List<List<double>> output = MatrixCalculus.Add(MatrixCalculus.Multiply(weightsHO, ActivationHiddenSigmoid), biasHO);
			List<List<double>> activationOutputSigmoid = MatrixCalculus.Sigmoid(output);
			return o;
			//return activationOutputSigmoid;
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

			var resultM = CreateMatrix.DenseOfRows(resultMatrix);
			var inputM = CreateMatrix.DenseOfRows(inputMatrix);
			//List<List<double>> guess = Predict(inputMatrix);
			var guessM = Predict(inputMatrix);

			//Hidden-Out
			var errM = resultM - guessM;
			var deltaWho = errM.GetDeltaError(guessM, LEARNING_RATE) * activationHidden2.Transpose();
			var deltaBho = errM.GetDeltaError(guessM, LEARNING_RATE);
			//List<List<double>> errorMatrix = MatrixCalculus.Substract(resultMatrix, guess);
			//List<List<double>> deltaWeightsHO = MatrixCalculus.Multiply(GetDeltaError(errorMatrix, guess), MatrixCalculus.Transpose(ActivationHiddenSigmoid));
			//List<List<double>> deltaBiasHO = GetDeltaError(errorMatrix, guess);

			//In-Hidden
			var errHiddenM = Who.Transpose() * errM;
			var deltaWih = errHiddenM.GetDeltaError(activationHidden2, LEARNING_RATE) * inputM.Transpose();
			var deltaBih = errHiddenM.GetDeltaError(activationHidden2, LEARNING_RATE);
			//List<List<double>> errorHiddenMatrix = MatrixCalculus.Multiply(MatrixCalculus.Transpose(weightsHO), errorMatrix);
			//List<List<double>> deltaWeightsIH = MatrixCalculus.Multiply(GetDeltaError(errorHiddenMatrix, ActivationHiddenSigmoid), MatrixCalculus.Transpose(inputMatrix));
			//List<List<double>> deltaBiasIH = GetDeltaError(errorHiddenMatrix, ActivationHiddenSigmoid);

			Who = Who + deltaWho;
			Wih = Wih + deltaWih;
			//weightsHO = MatrixCalculus.Add(weightsHO, deltaWeightsHO);
			//weightsIH = MatrixCalculus.Add(weightsIH, deltaWeightsIH);

			Bho = Bho + deltaBho;
			Bih = Bih + deltaBih;
			
			//biasHO = MatrixCalculus.Add(biasHO, deltaBiasHO);
			//biasIH = MatrixCalculus.Add(biasIH, deltaBiasIH);
		}
		#endregion

		#region Private Methods
		//private List<List<double>> GetDeltaError(List<List<double>> errorMatrix, List<List<double>> resultMatrix)
		//{
		//	List<List<double>> dSigmoid = MatrixCalculus.Map(resultMatrix, derivativeSigmoid);

		//	return MatrixCalculus.Multiply(
		//		MatrixCalculus.HadamarProduct(
		//			errorMatrix
		//		, dSigmoid)
		//		, LEARNING_RATE);
		//}


		#endregion
	}
}