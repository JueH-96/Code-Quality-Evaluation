import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, M = int(input[ptr]), int(input[ptr+1])
        ptr +=2
        res = 0
        for __ in range(N):
            V = int(input[ptr])
            P = int(input[ptr+1])
            ptr +=2
            if V >= P:
                res += V - P
        print(res)

if __name__ == "__main__":
    main()