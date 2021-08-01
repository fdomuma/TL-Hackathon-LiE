import pickle
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

def load_model():
    model_temp = pickle.load(open("XGB_Boost_complex_model.h5", 'rb'))
    return model_temp


def single_preprocess(week, checkout_price, base_price, center_id, meal_id):
    meal_info = pd.read_csv("meal_info.csv")
    fc_info = pd.read_csv("fulfilment_center_info.csv")

    discount = int(base_price)-int(checkout_price)

    #fc info
    temp_row = fc_info[fc_info["center_id"] == int(center_id)]
    temp_values = temp_row.values.tolist()[0]
    city_code, region_code, center_type, op_area = temp_values[1:]

    #meal info
    temp_row = meal_info[meal_info["meal_id"] == int(meal_id)]
    temp_values = temp_row.values.tolist()[0]
    category, cuisine = temp_values[1:]

    test_temp = pd.DataFrame({"week": [week], "center_id":[center_id], "meal_id": [meal_id],
                              "checkout_price": checkout_price, "city_code": city_code, "region_code": region_code,
                              "op_area": op_area, "discount": discount})

    cols = ['category_Biryani', 'category_Desert', 'category_Extras', 'category_Fish', 'category_Other Snacks',
            'category_Pasta', 'category_Pizza', 'category_Rice Bowl', 'category_Salad', 'category_Sandwich',
            'category_Seafood', 'category_Soup', 'category_Starters', 'cuisine_Indian', 'cuisine_Italian',
            'cuisine_Thai', 'center_type_TYPE_B', 'center_type_TYPE_C']

    for col in cols:
        if str(category) in col:
            test_temp[col] = 1
        elif str(cuisine) in col:
            test_temp[col] = 1
        elif str(center_type) in col:
            test_temp[col] =1
        else:
            test_temp[col] = 0

    return test_temp


def single_prediction(week, checkout_price, base_price, center_id, meal_id, model):
    input = single_preprocess(int(week), float(checkout_price), float(base_price), int(center_id), int(meal_id))
    results = model.predict(input)
    dict_temp ={"Demand": str(results[0])}

    return dict_temp


def price_comparison(week, checkout_range, base_price, center_id, meal_id, model):
    """
    returns various demand predictions based on different checkout prices

    :param input: without checkout price
    :param model:
    :param checkout_range: It is important to tell that this is an object from type range
    :return:
    """

    poss_price = []
    i_value = []
    range_value = []
    for i in checkout_range:
        price_temp = single_prediction(int(week), i, int(base_price), int(center_id), int(meal_id), model)
        temp_values = price_temp["Demand"]
        poss_price.append(float(temp_values))
        i_value.append(i)

    dict_temp = {}
    for j, i in enumerate(i_value):
        dict_temp[i] = poss_price[j]

    #plt.plot(i_value, poss_price)
    #plt.show()
    return dict_temp



# Init app
app = Flask(__name__)


model = load_model()

@app.route('/', methods=['GET'])
def choose_feature():
   return render_template('index.html')

@app.route('/single_prediction', methods=['GET'])
def upload_template_sp():
   return render_template('index_sp.html')

@app.route('/compare_prices', methods=['GET'])
def upload_template_cp():
   return render_template('index_cp.html')

@app.route('/single_prediction_results', methods = ['GET', 'POST'])
def predict_value():
   if request.method == 'POST':
       week = request.form["week"]
       checkout_price = request.form["checkout_price"]
       base_price = request.form["base_price"]
       center_id = request.form["center_id"]
       meal_id = request.form["meal_id"]
       return single_prediction(week, checkout_price, base_price, center_id, meal_id, model)

@app.route('/compare_prices_results', methods = ['GET', 'POST'])
def compare_prices():
   if request.method == 'POST':
       week = request.form["week"]
       checkout_price_min = request.form["checkout_price_min"]
       checkout_price_max = request.form["checkout_price_max"]
       checkout_price_step = request.form["checkout_price_step"]
       base_price = request.form["base_price"]
       center_id = request.form["center_id"]
       meal_id = request.form["meal_id"]
       prices_range = range(int(checkout_price_min), int(checkout_price_max), int(checkout_price_step))
       return price_comparison(week, prices_range, base_price, center_id, meal_id, model)

# Run server
if __name__ == '__main__':
    app.run(debug=True)