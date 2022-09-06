import pandas as pd
import openpyxl


df = pd.DataFrame([[11, 21, 31], [12, 22, 32], [31, 32, 33]],
                  index=['one', 'two', 'three'], columns=['a', 'b', 'c'])

print(df)
df.to_excel('data.xlsx', sheet_name='plant_feature_sensor')



#(green_id,	farm_id,	plant_loc,	rgb_path,	noir_path,	ndvi_path,	leaf_area_index,	created_at)df = pd.DataFrame([[11, 21, 31],],
#index=['-'], columns=['location', 'rgb_path', 'nir_path','ndvi_path','leaf Area','mean Ndvi','Std Ndvi','mean Rgb','std Rgb','Timeing'])