
import numpy as np
import matplotlib.pyplot as plt

arr1 = np.load("data/new/roundv2-2.npy", allow_pickle=True)
arr2 = np.load("data/new/roundv2-3.npy", allow_pickle=True)
arr3 = np.load("data/new/roundv2-4.npy", allow_pickle=True)
arr4 = np.load("data/new/roundv2-5.npy", allow_pickle=True)
arr5 = np.load("data/new/roundv2-6.npy", allow_pickle=True)
arr6 = np.load("data/new/roundv2-8.npy", allow_pickle=True)
arr7 = np.load("data/new/roundv2-9.npy", allow_pickle=True)
arr8 = np.load("data/new/roundv2-7_wrongrecord.npy", allow_pickle=True)
arr9 = np.load("data/new/roundv2-10.npy", allow_pickle=True)
arr10 = np.load("data/new/roundv2-11.npy", allow_pickle=True)
arr11 = np.load("data/new/roundv2-13.npy", allow_pickle=True)
arr12 = np.load("data/new/roundv2-15.npy", allow_pickle=True)
arr13 = np.load("data/new/roundv2-16.npy", allow_pickle=True)
arr14 = np.load("data/new/roundv2-17.npy", allow_pickle=True)


arr1.flatten
arr2.flatten
arr3.flatten
arr4.flatten
arr5.flatten
arr6.flatten
arr7.flatten
arr8.flatten
arr9.flatten
arr10.flatten
arr11.flatten
arr12.flatten
arr13.flatten
arr14.flatten

all = np.append(arr1, arr2)
all = np.append(all, arr3)
all = np.append(all, arr4)
all = np.append(all, arr5)
all = np.append(all, arr6)
all = np.append(all, arr7)
all = np.append(all, arr8)
all = np.append(all, arr9)
all = np.append(all, arr10)
all = np.append(all, arr11)
all = np.append(all, arr12)
all = np.append(all, arr13)
all = np.append(all, arr14)

all = all.reshape(len(all) // 12, 12)
gyro_array = all[:, [10]] # only check gyro on Z axis

threshold_pattern = 0.67
threshold_similarity = 0.97


def reference_round_finder(rounds):
    top_list = []
    for round in rounds:
        round_count = 0
        compare_round = round
        for round_again in rounds:
            simi = check_track_similarity(compare_round, round_again)
            if simi > 0.98:
                round_count += 1
        top_list.append(round_count)
    return top_list.index(max(top_list))

def get_x_and_y(rounds, reference_round_id):
    i = 0
    x_values = []
    y_values = []
    for round in rounds:
        compare_round = rounds[reference_round_id]
        simi = check_track_similarity(compare_round, round)
       
        if simi > threshold_similarity:
            #print(simi)
            x_coords = np.cumsum(np.cos(np.radians(round)))
            y_coords = np.cumsum(np.sin(np.radians(round)))

            x_correction = (x_coords[-1] - x_coords[0]) / len(x_coords)
            y_correction = (y_coords[-1] - y_coords[0]) / len(y_coords)

            x_coords_corrected = x_coords - np.arange(len(x_coords)) * x_correction
            y_coords_corrected = y_coords - np.arange(len(y_coords)) * y_correction

            x_coords_corrected = -x_coords_corrected # changed to - to mirror track
            y_coords_corrected = y_coords_corrected
            #plt.plot(x_coords_corrected, y_coords_corrected, label=f"Round {i + 1}")
            i+=1
            x_values.append(x_coords_corrected)
            y_values.append(y_coords_corrected)

    return x_values, y_values

def draw_tracks(x_values, y_values):
    i=0
    for x_value,y_value in zip(x_values, y_values):
        plt.plot(x_value, y_value, label=f"Round {i + 1}")
        i += 1

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{i} Rounds of Integrated and Scaled Gyro Data")
    plt.show()

def check_track_similarity(arr1, arr2):
    if len(arr1) < len(arr2):
        new_length = min(len(arr1), len(arr2))
        indices_new = np.linspace(0, len(arr2) - 1, new_length)
        arr2 = np.interp(indices_new, np.arange(len(arr2)), arr2)

    elif len(arr1) > len(arr2):
        new_length = min(len(arr1), len(arr2))
        indices_new = np.linspace(0, len(arr1) - 1, new_length)
        arr1 = np.interp(indices_new, np.arange(len(arr1)), arr1)

    #cos_sim = np.dot(arr2, arr1) / (np.linalg.norm(arr2) * np.linalg.norm(arr1)) #another way to compare similarity
    corr_coef = np.corrcoef(arr1, arr2)[0, 1]
    return(corr_coef)


def sliding_window_autocorr(data, window_size):
    n = len(data)
    autocorr = np.zeros(window_size)
    for i in range(window_size):
        autocorr[i] = np.sum(data[:n - i] * data[i:])
    return autocorr

def find_repeating_pattern(data, min_pattern_length=50, max_pattern_length=None, threshold=threshold_pattern):
    if max_pattern_length is None:
        max_pattern_length = len(data) // 2

    # Compute the autocorrelation with a sliding window
    autocorr = sliding_window_autocorr(data, max_pattern_length)

    # Normalize the autocorrelation
    autocorr /= autocorr[0]

    # Find the first peak that is greater than the threshold and has a minimum distance from the main peak (zero lag)
    for i in range(min_pattern_length, max_pattern_length):
        if autocorr[i] > threshold:
            return i
    return -1

def integrate_and_scale_data(data):
    integrated_data = np.cumsum(data)
    scale_factor = 360 / integrated_data[-1]
    scaled_data = integrated_data * scale_factor
    return scaled_data

def slice_data_into_rounds(data, pattern_length):
    rounds = []
    num_slices = len(data) // pattern_length
    for i in range(num_slices):
        start = i * pattern_length
        end = (i + 1) * pattern_length
        round_data = data[start:end]
        integrated_round_data = integrate_and_scale_data(round_data)
        rounds.append(integrated_round_data)

    return rounds

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def find_rotation_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_u, v2_u)
    cross_product = np.cross(v1_u, v2_u)
    angle = -np.arctan2(cross_product, dot_product)
    return angle

def rotate_points(x_values, y_values, pivot, angle):
    x_rotated = np.cos(angle) * (x_values - pivot[0]) - np.sin(angle) * (y_values - pivot[1]) + pivot[0]
    y_rotated = np.sin(angle) * (x_values - pivot[0]) + np.cos(angle) * (y_values - pivot[1]) + pivot[1]
    return x_rotated, y_rotated


def main():
    # Load the gyro data
    data = gyro_array

    angle_index = 20

    # Find the repeating pattern
    pattern_length = find_repeating_pattern(data)

    # Slice the data into rounds, integrate and scale the data, and normalize the starting point
    if pattern_length > 0:
        rounds = slice_data_into_rounds(data, pattern_length)
        print(f"Found {len(rounds)} rounds in the data.")
        reference_round_id = reference_round_finder(rounds)
        # Plot the normalized integrated and scaled rounds
        x_values, y_values = get_x_and_y(rounds, reference_round_id)
        x_values = np.array(x_values)
        y_values = np.array(y_values)

        master_x = x_values[2]  # The x values of the master shape (first shape)
        master_y = y_values[2]  # The y values of the master shape (first shape)

        pivot = np.array([master_x[0], master_y[0]])  # The pivot point (shared point)

        master_vector = np.array([master_x[angle_index] - master_x[0], master_y[angle_index] - master_y[0]])

        for i in range(1, len(x_values)):
            other_x = x_values[i]
            other_y = y_values[i]

            other_vector = np.array([other_x[angle_index] - other_x[0], other_y[angle_index] - other_y[0]])

            angle_diff = find_rotation_angle(master_vector, other_vector)

            
            rotated_x, rotated_y = rotate_points(other_x, other_y, pivot, angle_diff)
            x_values[i] = rotated_x
            y_values[i] = rotated_y
        for value in x_values:
            print(value[0],value[1],value[20],value[40],value[100])
            print("--------")

        draw_tracks(x_values,y_values)
        #draw_tracks(prev_x_values,prev_y_values)


    else:
        print("No repeating pattern found.")

if __name__ == "__main__":
    main()





