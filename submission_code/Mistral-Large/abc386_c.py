import sys

def can_transform(K, S, T):
    if K == 1:
        if S == T:
            return "Yes"
        if len(S) == len(T):
            # Check if one replacement can make S equal to T
            diff_count = 0
            for i in range(len(S)):
                if S[i] != T[i]:
                    diff_count += 1
                if diff_count > 1:
                    return "No"
            return "Yes"
        elif len(S) == len(T) + 1:
            # Check if one deletion can make S equal to T
            i, j = 0, 0
            while i < len(S) and j < len(T):
                if S[i] != T[j]:
                    i += 1
                    if i - j > 1:
                        return "No"
                else:
                    i += 1
                    j += 1
            return "Yes"
        elif len(S) == len(T) - 1:
            # Check if one insertion can make S equal to T
            i, j = 0, 0
            while i < len(S) and j < len(T):
                if S[i] != T[j]:
                    j += 1
                    if j - i > 1:
                        return "No"
                else:
                    i += 1
                    j += 1
            return "Yes"
        else:
            return "No"
    else:
        return "No"

def main():
    input = sys.stdin.read
    data = input().split()

    K = int(data[0])
    S = data[1]
    T = data[2]

    result = can_transform(K, S, T)
    print(result)

if __name__ == "__main__":
    main()