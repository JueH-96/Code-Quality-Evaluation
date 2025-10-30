import sys
data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1

 for test in range(T):
     N = int(data[index])
     K = int(data[index+1])
     index += 2

     cakes = []
     for i in range(N):
         X = int(data[index])
         Y = int(data[index+1])
         Z = int(data[index+2])
         cakes.append((X, Y, Z))
         index += 3

     active_indices = list(range(N))
     pos = {i: i for i in range(N)}  # cake index to position in active_indices

     total_sum = 0

     for _ in range(K):
         # Find for attr 0 (X)
         max_val0 = -1000000000000
         max_idx0 = -1
         second_val0 = -1000000000000
         second_idx0 = -1
         for cake_idx in active_indices:
             val = cakes[cake_idx][0]
             if val > max_val0:
                 second_val0 = max_val0
                 second_idx0 = max_idx0
                 max_val0 = val
                 max_idx0 = cake_idx
             elif val > second_val0:
                 second_val0 = val
                 second_idx0 = cake_idx
         S0 = max_val0 + second_val0

         # For attr 1 (Y)
         max_val1 = -1000000000000
         max_idx1 = -1
         second_val1 = -1000000000000
         second_idx1 = -1
         for cake_idx in active_indices:
             val = cakes[cake_idx][1]
             if val > max_val1:
                 second_val1 = max_val1
                 second_idx1 = max_idx1
                 max_val1 = val
                 max_idx1 = cake_idx
             elif val > second_val1:
                 second_val1 = val
                 second_idx1 = cake_idx
         S1 = max_val1 + second_val1

         # For attr 2 (Z)
         max_val2 = -1000000000000
         max_idx2 = -1
         second_val2 = -1000000000000
         second_idx2 = -1
         for cake_idx in active_indices:
             val = cakes[cake_idx][2]
             if val > max_val2:
                 second_val2 = max_val2
                 second_idx2 = max_idx2
                 max_val2 = val
                 max_idx2 = cake_idx
             elif val > second_val2:
                 second_val2 = val
                 second_idx2 = cake_idx
         S2 = max_val2 + second_val2

         # Choose the attribute with max S
         if S0 >= S1 and S0 >= S2:
             S_chosen = S0
             chosen_idx1 = max_idx0
             chosen_idx2 = second_idx0
         elif S1 >= S0 and S1 >= S2:
             S_chosen = S1
             chosen_idx1 = max_idx1
             chosen_idx2 = second_idx1
         else:
             S_chosen = S2
             chosen_idx1 = max_idx2
             chosen_idx2 = second_idx2

         total_sum += S_chosen

         # Remove the two cakes
         for idx in [chosen_idx1, chosen_idx2]:
             pos_idx = pos[idx]
             if pos_idx < len(active_indices) - 1:
                 last_cake_idx = active_indices[-1]
                 active_indices[pos_idx] = last_cake_idx
                 pos[last_cake_idx] = pos_idx
             # Remove the last element
             del active_indices[-1]
             # Remove from pos
             del pos[idx]

     # After K pairs, print the sum
     print(total_sum)