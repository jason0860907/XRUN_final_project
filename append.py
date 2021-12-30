import json

# hit_region = None
# with open('hit_region.json', 'r') as f:
#     hit_region = json.load(f)


# for key in hit_region.keys():
#     for k in hit_region[key].keys():
#         hit_region[key][k] = [hit_region[key][k]]
#     hit_region[key] = [hit_region[key]]
# hit_region = [hit_region]

# bounce_region = {"left": {"pos_neg": ["0", "0", "0", "0", "0"], "positive": ["0", "0", "0", "0", "0"], "negtive": ["0", "0", "0", "0", "0"], "ratio": ["0", "0"]}, "right": {"pos_neg": ["0", "0", "0", "0", "0"], "positive": ["0", "0", "0", "0", "0"], "negtive": ["0", "0", "0", "0", "0"], "ratio": ["0", "0"]}}
# try:
#     with open('bounce_region.json', 'r') as f:
#         bounce_region = json.load(f)
# except:
#     print('Cannot load bounce_region.json. Use default.')
# bounce_region = [bounce_region]

# region = [{'hit_region': hit_region, 'bounce_region': bounce_region}]

# old = None
# with open('台南大學_211012_1_3.json', 'r') as f:
#     old = json.load(f)
# idx = len(old['time'])
# old['time'].append({str(idx): region})  # append results

# with open('台南大學_211012_1_3_test.json', 'w') as f:
#     json.dump(old, f)
with open('testing.json', 'r') as f:
    final = json.load(f)
time = final['time']
print(len(time))
print(time[5]['5'])

with open('台南大學_211012_1_3.json', 'r') as f:
    final = json.load(f)
time = final['time']
print(len(time))
print(time[5]['5'])
