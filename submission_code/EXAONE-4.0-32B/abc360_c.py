import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    W = list(map(int, data[1+n:1+2*n]))
    
    total_weight = sum(W)
    max_in_box = [0] * (n + 1)
    
    for i in range(n):
        box_id = A[i]
        if W[i] > max_in_box[box_id]:
            max_in_box[box_id] = W[i]
            
    saving = sum(max_in_box[1:])
    answer = total_weight - saving
    print(answer)

if __name__ == "__main__":
    main()