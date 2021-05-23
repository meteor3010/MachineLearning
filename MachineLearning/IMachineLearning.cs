using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
	public interface IMachineLearning
	{
		void Train(IList<double> datas, IList<double> expectedValues);

		double[] Predict(IList<double> datas);
	}
}
