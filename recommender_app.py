import os
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from math import pi
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy import spatial
import ast
import re
from matplotlib import gridspec

def scale_value(value, min_value, max_value):
    value = np.clip(value, min_value, max_value)
    return (value - min_value) / (max_value - min_value)

def get_range_label(label_range_dict, value):
    for label, (lower, upper) in label_range_dict.items():
        if lower <= value <= upper:
            return label
    return None

def average_vector(food_items, model):
    vectors = [model[item] for item in food_items]
    return np.mean(vectors, axis=0)

def aroma_similarity(nonaroma, avg_food_vec, food_info):
    aroma_vec_str = food_info.loc[nonaroma, 'average_vec']
    aroma_vec = np.array(ast.literal_eval(re.sub('\s+', ',', aroma_vec_str).replace('[,', '[')))
    sim_score = 1 - spatial.distance.cosine(aroma_vec, avg_food_vec)
    scaled_sim_score = scale_value(sim_score, food_info.loc[nonaroma, 'farthest'], food_info.loc[nonaroma, 'closest'])
    return scaled_sim_score, get_range_label(food_weights[nonaroma], scaled_sim_score)

def compute_food_properties(food_items):
    avg_vec = average_vector(food_items, word_vectors)
    aroma_profiles = {aroma: aroma_similarity(aroma, avg_vec, food_info) for aroma in ['sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']}
    food_heaviness = aroma_similarity('weight', avg_vec, food_info)
    return aroma_profiles, food_heaviness, avg_vec

wine_data = pd.read_csv('wine_aromas_nonaromas.csv').set_index('Unnamed: 0')

for feature in ['weight', 'acid', 'salt', 'bitter']:
    wine_data[feature] = 1 - wine_data[feature]

descriptors = pd.read_csv('wine_variety_descriptors.csv').set_index('index')
word2vec = Word2Vec.load("food_word2vec_model.bin")
word_vectors = word2vec.wv
food_info = pd.read_csv('average_nonaroma_vectors.csv').set_index('Unnamed: 0')

food_weights = {
    'weight': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    'sweet': {1: (0, 0.45), 2: (0.45, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
    'acid': {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.7), 4: (0.7, 1)},
    'salt': {1: (0, 0.3), 2: (0.3, 0.55), 3: (0.55, 0.8), 4: (0.8, 1)},
    'piquant': {1: (0, 0.4), 2: (0.4, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
    'fat': {1: (0, 0.4), 2: (0.4, 0.5), 3: (0.5, 0.6), 4: (0.6, 1)},
    'bitter': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.65), 4: (0.65, 1)}
}
wine_weights = {
    'weight': {1: (0, 0.25), 2: (0.25, 0.45), 3: (0.45, 0.75), 4: (0.75, 1)},
    'sweet': {1: (0, 0.25), 2: (0.25, 0.6), 3: (0.6, 0.75), 4: (0.75, 1)},
    'acid': {1: (0, 0.05), 2: (0.05, 0.25), 3: (0.25, 0.5), 4: (0.5, 1)},
    'salt': {1: (0, 0.15), 2: (0.15, 0.25), 3: (0.25, 0.7), 4: (0.7, 1)},
    'piquant': {1: (0, 0.15), 2: (0.15, 0.3), 3: (0.3, 0.6), 4: (0.6, 1)},
    'fat': {1: (0, 0.25), 2: (0.25, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    'bitter': {1: (0, 0.2), 2: (0.2, 0.37), 3: (0.37, 0.6), 4: (0.6, 1)}
}

wine_data_normalized = wine_data.copy()
for feature, ranges in wine_weights.items():
    wine_data_normalized[feature] = wine_data_normalized[feature].apply(lambda x: get_range_label(ranges, x))
wine_data_normalized = wine_data_normalized.sort_index()
food_list = descriptors['descriptors'].unique()

def filter_by_weight(data, weight_bounds):
    """Select wines with a weight within a specified range."""
    criteria = (data['weight'] >= weight_bounds[1] - 1) & (data['weight'] <= weight_bounds[1])
    return data[criteria]

def filter_by_acidity(data, non_aroma_criteria):
    """Select wines with an acidity level greater than or equal to the food's."""
    return data[data['acid'] >= non_aroma_criteria['acid'][1]]

def filter_by_sweetness(data, non_aroma_criteria):
    """Select wines with a sweetness level greater than or equal to the food's."""
    return data[data['sweet'] >= non_aroma_criteria['sweet'][1]]

def exclude_bitter_wines(data, non_aroma_criteria):
    """Avoid wines that are too bitter when food is also bitter."""
    if non_aroma_criteria['bitter'][1] == 4:
        return data[data['bitter'] <= 2]
    return data

def avoid_bitter_and_salt(data, non_aroma_criteria):
    """Rule out wines when either wine is too salty and food is too bitter, or vice versa."""
    if non_aroma_criteria['bitter'][1] == 4:
        data = data[data['salt'] <= 2]
    if non_aroma_criteria['salt'] == 4:
        data = data[data['bitter'][1] <= 2]
    return data

def avoid_acid_and_bitter(data, non_aroma_criteria):
    """Rule out wines that are either too acidic with bitter food, or too bitter with acidic food."""
    if non_aroma_criteria['acid'][1] == 4:
        data = data[data['bitter'] <= 2]
    if non_aroma_criteria['bitter'][1] == 4:
        data = data[data['acid'] <= 2]
    return data

def avoid_acid_and_piquant(data, non_aroma_criteria):
    """Rule out wines that are either too acidic with piquant food, or too piquant with acidic food."""
    if non_aroma_criteria['acid'][1] == 4:
        data = data[data['piquant'] <= 2]
    if non_aroma_criteria['piquant'][1] == 4:
        data = data[data['acid'] <= 2]
    return data

def apply_non_aroma_filters(wine_data, non_aroma_criteria, weight_bounds):
    """Apply all the filtering rules sequentially to the wine dataset."""
    data = filter_by_weight(wine_data, weight_bounds)

    filters = [
        filter_by_acidity, filter_by_sweetness, exclude_bitter_wines,
        avoid_bitter_and_salt, avoid_acid_and_bitter, avoid_acid_and_piquant
    ]

    for filter_func in filters:
        potential_selection = filter_func(data, non_aroma_criteria)
        if len(potential_selection) > 5:
            data = potential_selection

    return data

def match_sweetness(data, flavor_profiles):
    """Determine if sweet food matches with contrasting wine flavors."""
    if flavor_profiles['sweet'][1] == 4:
        conditions = (data.bitter == 4) | (data.fat == 4) | (data.piquant == 4) | (data.salt == 4) | (data.acid == 4)
        data['match_type'] = np.where(conditions, 'contrasting', data.match_type)
    return data

def match_acidity(data, flavor_profiles):
    """Determine if acidic food matches with contrasting wine flavors."""
    if flavor_profiles['acid'][1] == 4:
        conditions = (data.sweet == 4) | (data.fat == 4) | (data.salt == 4)
        data['match_type'] = np.where(conditions, 'contrasting', data.match_type)
    return data

def match_saltiness(data, flavor_profiles):
    """Determine if salty food matches with contrasting wine flavors."""
    if flavor_profiles['salt'][1] == 4:
        conditions = (data.bitter == 4) | (data.sweet == 4) | (data.piquant == 4) | (data.fat == 4) | (data.acid == 4)
        data['match_type'] = np.where(conditions, 'contrasting', data.match_type)
    return data

def match_piquancy(data, flavor_profiles):
    """Determine if piquant food matches with contrasting wine flavors."""
    if flavor_profiles['piquant'][1] == 4:
        conditions = (data.sweet == 4) | (data.fat == 4) | (data.salt == 4)
        data['match_type'] = np.where(conditions, 'contrasting', data.match_type)
    return data

def match_fattiness(data, flavor_profiles):
    """Determine if fatty food matches with contrasting wine flavors."""
    if flavor_profiles['fat'][1] == 4:
        conditions = (data.bitter == 4) | (data.sweet == 4) | (data.piquant == 4) | (data.salt == 4) | (data.acid == 4)
        data['match_type'] = np.where(conditions, 'contrasting', data.match_type)
    return data

def match_bitterness(data, flavor_profiles):
    """Determine if bitter food matches with contrasting wine flavors."""
    if flavor_profiles['bitter'][1] == 4:
        conditions = (data.sweet == 4) | (data.fat == 4) | (data.salt == 4)
        data['match_type'] = np.where(conditions, 'contrasting', data.match_type)
    return data

def find_congruent(pair_type, max_flavor_value, wine_flavor_value):
    if pair_type == 'congruent':
        return 'congruent'
    elif wine_flavor_value >= max_flavor_value:
        return 'congruent'
    else:
        return ''

def determine_pairing_type(data, flavor_profiles):
    """Identify congruent or contrasting pairings based on wine and food flavors."""
    # Identify congruent match
    highest_flavor_value = max([val[1] for val in flavor_profiles.values()])
    dominant_flavors = [k for k, v in flavor_profiles.items() if v[1] == highest_flavor_value]

    data['match_type'] = ''
    for flavor in dominant_flavors:
        data['match_type'] = data.apply(lambda x: find_congruent(x['match_type'], flavor_profiles[flavor][1], x[flavor]), axis=1)

    # Identify contrasting matches
    pairing_tests = [match_sweetness, match_acidity, match_saltiness, match_piquancy, match_fattiness, match_bitterness]
    for test in pairing_tests:
        data = test(data, flavor_profiles)

    return data

def rearrange_by_aroma_proximity(dataframe, food_scent):

    def convert_nparray_string(array_str):
        formatted_str = re.sub('\s+', ',', array_str).replace('[,', '[')
        return np.array(ast.literal_eval(formatted_str))

    dataframe['aroma'] = dataframe['aroma'].apply(convert_nparray_string)
    dataframe['scent_difference'] = dataframe['aroma'].apply(lambda x: spatial.distance.cosine(x, food_scent))
    dataframe.sort_values(by=['scent_difference'], ascending=True, inplace=True)

    return dataframe

def calculate_descriptor_similarity(descriptor, food_vector):
    if descriptor in word_vectors:
        descriptor_vector = word_vectors[descriptor]
        similarity_score = 1 - spatial.distance.cosine(descriptor_vector, food_vector)
    else:
        similarity_score = 0
    return similarity_score

def find_top_descriptors(suggestion):
    descriptor_stats = descriptors.filter(like=suggestion, axis=0)
    descriptor_stats_copy = descriptor_stats.copy()
    descriptor_stats_copy['similarity'] = descriptor_stats['descriptors'].apply(lambda x: calculate_descriptor_similarity(x, scent_vector))
    descriptor_stats = descriptor_stats_copy.copy()
    descriptor_stats_copy = descriptor_stats.copy()
    descriptor_stats_copy.sort_values(by=['similarity', 'relative_frequency'], ascending=False, inplace=True)
    descriptor_stats = descriptor_stats_copy.copy()
    descriptor_stats.sort_values(by=['similarity', 'relative_frequency'], ascending=False, inplace=True)
    top_descriptors = descriptor_stats.head(5)
    influential_descriptors = list(top_descriptors['descriptors'])
    return influential_descriptors

def get_pairing_details(wine_suggestions, full_taste_profile, type_of_pairing):
    selected_pairings = wine_suggestions[wine_suggestions['match_type'] == type_of_pairing].head(4)
    wine_list = selected_pairings.index.tolist()
    taste_profiles = full_taste_profile.loc[wine_list]
    taste_attributes = taste_profiles[['sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']].to_dict('records')
    wine_body_weights = taste_profiles['weight'].tolist()
    dominant_descriptors = selected_pairings['key_descriptors'].tolist()
    return wine_list, taste_attributes, wine_body_weights, dominant_descriptors

def generate_radar_chart(grid_layout, chart_index, data_points, chart_title, line_color, match_type):

    # Count of variables
    taste_categories = list(dish_non_aromatics.keys())
    total_categories = len(taste_categories)

    # Calculate angle for each axis in the plot
    sector_angles = [i / float(total_categories) * 2 * pi for i in range(total_categories)]
    sector_angles += sector_angles[:1]

    # Set up the radar chart
    chart_area = plt.subplot(grid_layout[chart_index], polar=True)

    # Adjust the orientation
    chart_area.set_theta_offset(pi / 2)
    chart_area.set_theta_direction(-1)

    # Label axes
    plt.xticks(sector_angles[:-1], taste_categories, color='grey', size=11)

    # Add y-axis labels
    chart_area.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25","0.50","0.75", "1.00"], color="grey", size=0)
    plt.ylim(0, 1)

    # Plot the data
    taste_values = list(data_points.values())
    taste_values += taste_values[:1]
    chart_area.plot(sector_angles, taste_values, color=line_color, linewidth=2, linestyle='solid')
    chart_area.fill(sector_angles, taste_values, color=line_color, alpha=0.4)
    #chart_area.grid(False)
    #chart_area.spines['polar'].set_visible(False)

    # Split and format the title if required
    title_parts = str(chart_title).split(',')
    formatted_title = []
    for idx, phrase in enumerate(title_parts):
        if (idx % 2) == 0 and idx > 0:
            reformatted_phrase = '\n' + phrase.strip()
            formatted_title.append(reformatted_phrase)
        else:
            reformatted_phrase = phrase.strip()
            formatted_title.append(reformatted_phrase)
    title_with_type = ', '.join(formatted_title) + '\n' + '(' + str(match_type) + ')'

    plt.title(title_with_type, size=13, color='black', y=1.2)

def draw_intensity_line(grid_layout, chart_index, intensity_val, point_color):
    chart_area = plt.subplot(grid_layout[chart_index])
    chart_area.set_xlim(-1, 2)
    chart_area.set_ylim(0, 3)

    # Create horizontal line
    start_x = 0
    end_x = 1
    y_position = 1
    line_thickness = 0.2

    plt.hlines(y_position, start_x, end_x)
    plt.vlines(start_x, y_position - line_thickness / 2., y_position + line_thickness / 2.)
    plt.vlines(end_x, y_position - line_thickness / 2., y_position + line_thickness / 2.)

    # Place a point on the line
    point_x = intensity_val
    plt.plot(point_x, y_position, 'ko', ms=10, mfc=point_color)

    # Add end-labels
    plt.text(start_x - 0.1, y_position, 'Mild-Intensity', horizontalalignment='right', fontsize=11, color='grey')
    plt.text(end_x + 0.1, y_position, 'High-Intensity', horizontalalignment='left', fontsize=11, color='grey')
    # chart_area.grid(False)
    # chart_area.spines['polar'].set_visible(False)
    plt.axis('off')

def display_important_notes(grid_layout, chart_index, significant_notes):
    notes_area = plt.subplot(grid_layout[chart_index])

    notes_area.set_xticks([])
    notes_area.set_yticks([])
    for edge in notes_area.spines.values():
        edge.set_visible(False)
    notes_area.invert_yaxis()

    notes_text = f'Highlighted wine aromas:\n\n{significant_notes[0]}, \n{significant_notes[1]}, \n{significant_notes[2]}, \n{significant_notes[3]}, \n{significant_notes[4]}'
    notes_area.text(x=0, y=1, s=notes_text, fontsize=12, color='grey')

def present_wine_suggestions(suggested_wines, wine_non_aromatics, wine_intensity, note_highlights, match_methods):

    total_rows = 3
    total_columns = 4
    plt.figure(figsize=(20, 7), dpi=96)

    layout = gridspec.GridSpec(3, 4, height_ratios=[3, 0.5, 1])

    radar_chart_index = 0
    intensity_line_index = 4
    notes_index = 8
    colors = ['black', 'blue', 'purple', 'brown']
    for w in range(4):
        generate_radar_chart(layout, radar_chart_index, wine_non_aromatics[w], suggested_wines[w], colors[w], match_methods[w])
        draw_intensity_line(layout, intensity_line_index, wine_intensity[w], point_color=colors[w])
        display_important_notes(layout, notes_index, note_highlights[w])
        radar_chart_index += 1
        intensity_line_index += 1
        notes_index += 1

def recommend(ingredients):
    sweet_course = ingredients

    dish_non_aromatics, dish_importance, scent_vector = compute_food_properties(sweet_course)

    suggested_wines = wine_data_normalized.copy()
    
    suggested_wines = apply_non_aroma_filters(suggested_wines, dish_non_aromatics, dish_importance)

    suggested_wines = determine_pairing_type(suggested_wines, dish_non_aromatics)
    
    suggested_wines = rearrange_by_aroma_proximity(suggested_wines, scent_vector)
    # print(suggested_wines)

    # suggested_wines['key_descriptors'] = suggested_wines.index.map(find_top_descriptors)
    
    

    # Check for available contrasting wine recommendations
    # try:
    #     opposing_wines, opposing_non_aromatics, opposing_wine_strength, dominant_notes_opposing = get_pairing_details(suggested_wines, wine_data, 'contrasting')
    # except Exception as e:
    #     print("Opposing ----------------")
    #     print(e)
    #     opposing_wines = []

    # # Check for available congruent wine recommendations
    # try:
    #     matching_wines, matching_non_aromatics, matching_wine_strength, dominant_notes_matching = get_pairing_details(suggested_wines, wine_data, 'congruent')
    # except Exception as e:
    #     print("Matching --------")
    #     print(e)
    #     matching_wines = []

    # # Offer a mix of opposing and matching wines, if available
    # if len(opposing_wines) >= 2 and len(matching_wines) >= 2:
    #     selected_wines = opposing_wines[:2] + matching_wines[:2]
    #     wine_non_aromatics = opposing_non_aromatics[:2] + matching_non_aromatics[:2]
    #     wine_intensity = opposing_wine_strength[:2] + matching_wine_strength[:2]
    #     main_notes = dominant_notes_opposing[:2] + dominant_notes_matching[:2]
    #     wine_pairing_styles = ['Contrasting', 'Contrasting', 'Congruent', 'Congruent']
    # elif len(opposing_wines) >= 2:
    #     selected_wines = opposing_wines
    #     wine_non_aromatics = opposing_non_aromatics
    #     wine_intensity = opposing_wine_strength
    #     main_notes = dominant_notes_opposing
    #     wine_pairing_styles = ['Contrasting', 'Contrasting', 'Contrasting', 'Contrasting']
    # else:
    #     selected_wines = matching_wines
    #     wine_non_aromatics = matching_non_aromatics
    #     wine_intensity = matching_wine_strength
    #     main_notes = dominant_notes_matching
    #     wine_pairing_styles = ['Congruent', 'Congruent', 'Congruent', 'Congruent']
    # # for wine in selected_wines:
    # #     st.header(wine)
    # # plt.figure(figsize=(4, 5), dpi=75)
    # # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 0.5])

    # # dish_non_aromatics_norm = {k: v[0] for k, v in dish_non_aromatics.items()}
    # # food_names = ' + '.join(sweet_course)
    # # generate_radar_chart(gs, 0, dish_non_aromatics_norm, 'Selected Foods Profile:', 'green', food_names)
    # # draw_intensity_line(gs, 1, dish_importance[0], point_color='green')

    # # plt.tight_layout()

    # print(suggested_wines)
    print(suggested_wines.columns)
    return suggested_wines[:5]
# Streamlit app


# st.title("Wine Pairing Recommendation App")
# description = """
# ## Culinary Delights & Wine Pairing 🍷
# Culinary delights and wine share a storied tradition of harmonious pairing, each elevating the experience of the other. 

# **Embark on a journey of sensory pleasure** with our Wine Pairing Recommendation App. Simply select the dishes you plan to savor, and allow our intelligent algorithm to recommend wines that perfectly complement your culinary choices.

# Whether you're seeking congruent pairings that mirror the flavor profiles of your dishes or contrasting pairings that offer a delightful counterpoint, we've got you covered. **Begin your gourmet adventure below!**
# """

# st.markdown(description)

# # Multi-select dropdown for selecting food items
# selected_foods = st.multiselect("Select food items:", food_list)

# # Check if selected_options is empty and show a message
# if not selected_foods:
#     st.warning("Please select at least one option.")
# else:
#             # Button to recommend wines
#             if st.button("Recommend"):
#                 # Perform wine pairing recommendation based on selected foods
#                 sweet_course = selected_foods
            
#                 dish_non_aromatics, dish_importance, scent_vector = compute_food_properties(sweet_course)
                
#                 suggested_wines = wine_data_normalized.copy()
#                 suggested_wines = apply_non_aroma_filters(suggested_wines, dish_non_aromatics, dish_importance)
#                 suggested_wines = determine_pairing_type(suggested_wines, dish_non_aromatics)
#                 suggested_wines = rearrange_by_aroma_proximity(suggested_wines, scent_vector)
#                 suggested_wines['key_descriptors'] = suggested_wines.index.map(find_top_descriptors)
                
#                 # Check for available contrasting wine recommendations
#                 try:
#                     opposing_wines, opposing_non_aromatics, opposing_wine_strength, dominant_notes_opposing = get_pairing_details(suggested_wines, wine_data, 'contrasting')
#                 except Exception as e:
#                     print("Opposing ----------------")
#                     print(e)
#                     opposing_wines = []
                
#                 # Check for available congruent wine recommendations
#                 try:
#                     matching_wines, matching_non_aromatics, matching_wine_strength, dominant_notes_matching = get_pairing_details(suggested_wines, wine_data, 'congruent')
#                 except Exception as e:
#                     print("Matching --------")
#                     print(e)
#                     matching_wines = []
                
#                 # Offer a mix of opposing and matching wines, if available
#                 if len(opposing_wines) >= 2 and len(matching_wines) >= 2:
#                     selected_wines = opposing_wines[:2] + matching_wines[:2]
#                     wine_non_aromatics = opposing_non_aromatics[:2] + matching_non_aromatics[:2]
#                     wine_intensity = opposing_wine_strength[:2] + matching_wine_strength[:2]
#                     main_notes = dominant_notes_opposing[:2] + dominant_notes_matching[:2]
#                     wine_pairing_styles = ['Contrasting', 'Contrasting', 'Congruent', 'Congruent']
#                 elif len(opposing_wines) >= 2:
#                     selected_wines = opposing_wines
#                     wine_non_aromatics = opposing_non_aromatics
#                     wine_intensity = opposing_wine_strength
#                     main_notes = dominant_notes_opposing
#                     wine_pairing_styles = ['Contrasting', 'Contrasting', 'Contrasting', 'Contrasting']
#                 else:
#                     selected_wines = matching_wines
#                     wine_non_aromatics = matching_non_aromatics
#                     wine_intensity = matching_wine_strength
#                     main_notes = dominant_notes_matching
#                     wine_pairing_styles = ['Congruent', 'Congruent', 'Congruent', 'Congruent']
#                 # for wine in selected_wines:
#                 #     st.header(wine)
#                 plt.figure(figsize=(4, 5), dpi=75)
#                 gs = gridspec.GridSpec(2, 1, height_ratios=[3, 0.5])
                
#                 dish_non_aromatics_norm = {k: v[0] for k, v in dish_non_aromatics.items()}
#                 food_names = ' + '.join(sweet_course)
#                 generate_radar_chart(gs, 0, dish_non_aromatics_norm, 'Selected Foods Profile:', 'green', food_names)
#                 draw_intensity_line(gs, 1, dish_importance[0], point_color='green')
                
#                 plt.tight_layout()
                
#                 st.pyplot(plt.show())
#                 st.pyplot(present_wine_suggestions(selected_wines, wine_non_aromatics, wine_intensity, main_notes, wine_pairing_styles))
    
# recommend(['beans'])