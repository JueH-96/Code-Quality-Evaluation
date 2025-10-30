import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    pos_dict = {}
    for idx, num in enumerate(A):
        if num not in pos_dict:
            pos_dict[num] = []
        pos_dict[num].append(idx)
    
    L = []
    curr = -1
    for i in range(m):
        x = B[i]
        if x not in pos_dict:
            print("No")
            return
        arr = pos_dict[x]
        j = bisect.bisect_left(arr, curr + 1)
        if j == len(arr):
            print("No")
            return
        L.append(arr[j])
        curr = arr[j]
        
    R = [0] * m
    curr = 10**9
    for i in range(m-1, -1, -1):
        x = B[i]
        arr = pos_dict[x]
        j = bisect.bisect_left(arr, curr)
        if j == 0:
            print("No")
            return
        chosen_index = arr[j-1]
        R[i] = chosen_index
        curr = chosen_index
        
    if L == R:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()