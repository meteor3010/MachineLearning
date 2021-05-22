using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
	public interface IMachineLearning
	{

		void Train(List<double> datas, List<double> expectedValues);

		void Train(List<List<double>> dataSet, List<List<double>> expectedValues);

		List<double> Predict(List<double> datas);
	}
}
