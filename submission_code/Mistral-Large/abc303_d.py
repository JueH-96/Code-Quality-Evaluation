import sys

def min_time_to_type_string(X, Y, Z, S):
    n = len(S)
    min_time = 0
    caps_lock_on = False

    for i in range(n):
        if S[i] == 'A':
            if not caps_lock_on:
                min_time += min(X + Z, Y)
                caps_lock_on = True
            else:
                min_time += X
        else:  # S[i] == 'a'
            if caps_lock_on:
                min_time += min(X + Z, Y)
                caps_lock_on = False
            else:
                min_time += X

    return min_time

def main():
    input = sys.stdin.read
    data = input().split()

    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3]

    result = min_time_to_type_string(X, Y, Z, S)
    print(result)

if __name__ == "__main__":
    main()