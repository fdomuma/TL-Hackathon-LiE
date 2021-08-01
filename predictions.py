import pickle
import pandas as pd


def load_model():
    model_temp = pickle.load(open("model.h5", 'rb'))
    return model_temp


def single_preprocess(week, checkout_price, base_price, center, meal):
    test_temp = pd.DataFrame({"week": [week], "checkout_price": [checkout_price], "base_price": [base_price]})
    cols = ["x0_1062", "x0_1109", "x0_1198", "x0_1207", "x0_1216", "x0_1230", "x0_1247", "x0_1248", "x0_1311",
            "x0_1438",
            "x0_1445", "x0_1525", "x0_1543", "x0_1558", "x0_1571", "x0_1727", "x0_1754", "x0_1770", "x0_1778",
            "x0_1803",
            "x0_1847", "x0_1878", "x0_1885", "x0_1902", "x0_1962", "x0_1971", "x0_1993", "x0_2104", "x0_2126",
            "x0_2139",
            "x0_2290", "x0_2304", "x0_2306", "x0_2322", "x0_2444", "x0_2490", "x0_2492", "x0_2494", "x0_2539",
            "x0_2569",
            "x0_2577", "x0_2581", "x0_2631", "x0_2640", "x0_2664", "x0_2704", "x0_2707", "x0_2760", "x0_2826",
            "x0_2867",
            "x0_2956", "x1_10", "x1_11", "x1_13", "x1_14", "x1_17", "x1_20", "x1_23", "x1_24", "x1_26", "x1_27",
            "x1_29",
            "x1_30", "x1_32", "x1_34", "x1_36", "x1_39", "x1_41", "x1_42", "x1_43", "x1_50", "x1_51", "x1_52", "x1_53",
            "x1_55", "x1_57", "x1_58", "x1_59", "x1_61", "x1_64", "x1_65", "x1_66", "x1_67", "x1_68", "x1_72", "x1_73",
            "x1_74", "x1_75", "x1_76", "x1_77", "x1_80", "x1_81", "x1_83", "x1_86", "x1_88", "x1_89", "x1_91", "x1_92",
            "x1_93", "x1_94", "x1_97", "x1_99", "x1_101", "x1_102", "x1_104", "x1_106", "x1_108", "x1_109", "x1_110",
            "x1_113", "x1_124", "x1_126", "x1_129", "x1_132", "x1_137", "x1_139", "x1_143", "x1_145", "x1_146",
            "x1_149",
            "x1_152", "x1_153", "x1_157", "x1_161", "x1_162", "x1_174", "x1_177", "x1_186"]

    for col in cols:
        if str(center) in col:
            test_temp[col] = 1
        elif str(meal) in col:
            test_temp[col] = 1
        else:
            test_temp[col] = 0

    return test_temp


def single_prediction(input, model):
    results = model.predict(input)
    return results


model = load_model()
test = single_preprocess(146, 125, 184.9, 55, 1885)
print(test)

possPrices = []
for i in range(125, 175, 5):
    preprocessed_data = single_preprocess(146, i, 184.9, 55, 1885)
    possPrices.append(single_prediction(preprocessed_data, model))

#prediction = single_prediction(test, model)


print(possPrices)
