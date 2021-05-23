using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.Calculus
{
	public static class MatrixExtentions
	{
		public static Matrix<double> GetDeltaError(this Matrix<double> errorMatrix, Matrix<double> resultMatrix, double learningRate)
		{
			var matrix = resultMatrix.Map(derivativeSigmoid);
			return errorMatrix.PointwiseMultiply(matrix) * learningRate;
		}

		private static double derivativeSigmoid(double number)
		{
			return number * (1 - number);
		}
	}
}
