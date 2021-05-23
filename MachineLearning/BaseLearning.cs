using MachineLearning.Calculus;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
	public abstract class BaseLearning : IMachineLearning
	{
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

		public abstract void Train(IList<double> datas, IList<double> expectedValues);
		
		internal abstract Matrix<double> Predict(IEnumerable<IEnumerable<double>> data);
	}
}
