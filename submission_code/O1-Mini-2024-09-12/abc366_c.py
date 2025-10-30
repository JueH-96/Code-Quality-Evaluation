import sys

def main():
    data = sys.stdin.read().split()
    Q = int(data[0])
    counts = [0] * (10**6 + 1)
    num_diff = 0
    res = []
    idx = 1
    for _ in range(Q):
        if data[idx] == '1':
            x = int(data[idx +1])
            if counts[x] == 0:
                num_diff +=1
            counts[x] +=1
            idx +=2
        elif data[idx] == '2':
            x = int(data[idx +1])
            counts[x] -=1
            if counts[x] ==0:
                num_diff -=1
            idx +=2
        else:
            res.append(str(num_diff))
            idx +=1
    print('
'.join(res))

if __name__ == "__main__":
    main()