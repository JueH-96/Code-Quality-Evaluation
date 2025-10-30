# YOUR CODE HERE
import sys
import sys
import sys

def main():
    import sys
    import sys
    from collections import defaultdict

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    cuboids = []
    idx = 1
    for i in range(N):
        x1 = int(data[idx])
        y1 = int(data[idx+1])
        z1 = int(data[idx+2])
        x2 = int(data[idx+3])
        y2 = int(data[idx+4])
        z2 = int(data[idx+5])
        cuboids.append((x1, y1, z1, x2, y2, z2))
        idx +=6

    counts = [0]*N

    for axis in range(3):
        # axis =0: x, 1:y, 2:z
        # For each cuboid, get axis1 and axis2
        axis1_list = []
        axis2_list = []
        other1_list = []
        other2_list = []
        other3_list = []
        for i, cub in enumerate(cuboids):
            if axis ==0:
                axis1 = cub[0]
                axis2 = cub[3]
                other1 = cub[1]
                other2 = cub[2]
                other3 = cub[4:]
            elif axis ==1:
                axis1 = cub[1]
                axis2 = cub[4]
                other1 = cub[0]
                other2 = cub[2]
                other3 = (cub[3], cub[5])
            else:
                axis1 = cub[2]
                axis2 = cub[5]
                other1 = cub[0]
                other2 = cub[1]
                other3 = (cub[3], cub[4])
            axis1_list.append(axis1)
            axis2_list.append(axis2)
            other1_list.append(other1)
            other2_list.append(other2)
            other3_list.append(other3)
        
        # Create mapping from c to list A and list B
        # list A: axis2 ==c
        # list B: axis1 ==c
        A_map = defaultdict(list)
        B_map = defaultdict(list)
        for i in range(N):
            c_end = axis2_list[i]
            c_start = axis1_list[i]
            A_map[c_end].append( (other1_list[i], other2_list[i], other3_list[i], i))
            B_map[c_start].append( (other1_list[i], other2_list[i], other3_list[i], i))
        
        # Iterate through all possible c
        for c in range(101):
            list_A = A_map.get(c, [])
            list_B = B_map.get(c, [])
            if not list_A or not list_B:
                continue
            # Sort list_A and list_B by other1 (y or x or x etc.)
            list_A_sorted = sorted(list_A, key=lambda x: x[0])
            list_B_sorted = sorted(list_B, key=lambda x: x[0])
            # Sweep y
            j =0
            len_B = len(list_B_sorted)
            for a in list_A_sorted:
                a_o1, a_o2, a_o3, a_id = a
                # Move j to the first B where B.o2 > a.o1
                while j < len_B and list_B_sorted[j][1] <= a_o1:
                    j +=1
                k = j
                while k < len_B and list_B_sorted[k][0] < a_o2:
                    b_o1, b_o2, b_o3, b_id = list_B_sorted[k]
                    # Check z overlap
                    if a_o3[0] < b_o3[1] and a_o3[1] > b_o3[0]:
                        counts[a_id] +=1
                        counts[b_id] +=1
                    k +=1

    for cnt in counts:
        print(cnt)

if __name__ == "__main__":
    main()