# YOUR CODE HERE
import sys
import sys
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    T = int(data[ptr])
    ptr +=1
    results = []
    for _ in range(T):
        N = int(data[ptr])
        M = int(data[ptr+1])
        ptr +=2
        valid_boxes = []
        for _ in range(N):
            V = int(data[ptr])
            P = int(data[ptr+1])
            ptr +=2
            if V > P:
                profit = V - P
                valid_boxes.append((profit, V, P))
        # Sort the boxes by profit descending
        valid_boxes.sort(reverse=True, key=lambda x: x[0])
        # Take top M boxes
        selected = valid_boxes[:M]
        sum_V = sum(box[1] for box in selected)
        sum_P = sum(box[2] for box in selected)
        net = sum_V - sum_P
        results.append(str(net))
    print('
'.join(results))

if __name__ == "__main__":
    main()