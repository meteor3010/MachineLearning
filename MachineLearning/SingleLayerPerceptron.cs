using MachineLearning.Calculus;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;

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
		private Matrix<double> WeightIH;
		private Matrix<double> BiaisIH;
		private Matrix<double> WeightHO;
		private Matrix<double> BiaisHO;
		private Matrix<double> ActivationHidenSigmoid;
		#endregion

		#region Ctor
		public SingleLayerPerceptron(int numberInput, int nNeurons, int numberOutput)
		{
			NumberInput = numberInput;
			NumberNeurons = nNeurons;
			NumberOutput = numberOutput;
			WeightIH = CreateMatrix.Random<double>(NumberNeurons, NumberInput, new ContinuousUniform(0, 0.1));
			BiaisIH = CreateMatrix.Random<double>(NumberNeurons, 1, new ContinuousUniform(0, 0.1));

			WeightHO = CreateMatrix.Random<double>(NumberOutput, NumberNeurons, new ContinuousUniform(0, 0.1));
			BiaisHO = CreateMatrix.Random<double>(NumberOutput, 1, new ContinuousUniform(0, 0.1));
		}
		#endregion

		#region Public Methods
		public double[] Predict(IList<double> datas)
		{
			var matrix = datas.ToMatrix();
			var result = Predict(matrix);
			var array = result.ToArray();
			double[] ar = new double[array.Length];
			int i = 0;
			foreach (var arr in array)
			{
				ar[i++] = arr;
			}
			return ar;
		}

		public void Train(IList<double> datas, IList<double> expectedValues)
		{
			var resultMatrix = expectedValues.ToMatrix();
			var inputMatrix = datas.ToMatrix();

			var resultM = CreateMatrix.DenseOfRows(resultMatrix);
			var inputM = CreateMatrix.DenseOfRows(inputMatrix);
			var guessM = Predict(inputMatrix);

			//Hidden-Out
			var errM = resultM - guessM;
			var deltaWho = errM.GetDeltaError(guessM, LEARNING_RATE) * ActivationHidenSigmoid.Transpose();
			var deltaBho = errM.GetDeltaError(guessM, LEARNING_RATE);

			//In-Hidden
			var errHiddenM = WeightHO.Transpose() * errM;
			var deltaWih = errHiddenM.GetDeltaError(ActivationHidenSigmoid, LEARNING_RATE) * inputM.Transpose();
			var deltaBih = errHiddenM.GetDeltaError(ActivationHidenSigmoid, LEARNING_RATE);

			WeightHO = WeightHO + deltaWho;
			WeightIH = WeightIH + deltaWih;

			BiaisHO = BiaisHO + deltaBho;
			BiaisIH = BiaisIH + deltaBih;
		}
		#endregion

		#region Internal Methods
		internal Matrix<double> Predict(IEnumerable<IEnumerable<double>> data)
		{
			var Datas = CreateMatrix.DenseOfRows(data);
			ActivationHidenSigmoid = WeightIH.Multiply(Datas) + BiaisIH;

			ActivationHidenSigmoid = ActivationHidenSigmoid.Map(Tools.Sigmoid);

			var o = WeightHO.Multiply(ActivationHidenSigmoid) + BiaisHO;
			o = o.Map(Tools.Sigmoid);
			return o;
		}
		#endregion
	}
}