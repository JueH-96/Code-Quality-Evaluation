def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        M = int(input[ptr+1])
        ptr += 2
        max_gain = 0
        for __ in range(N):
            V = int(input[ptr])
            P = int(input[ptr+1])
            ptr += 2
            if M == 1:
                gain = V - P
            else:
                gain = 1 - P
            if gain > max_gain:
                max_gain = gain
        print(max_gain if max_gain > 0 else 0)

if __name__ == '__main__':
    main()