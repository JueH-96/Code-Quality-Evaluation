# YOUR CODE HERE
def minimum_toll(S_x, S_y, T_x, T_y):
    if (S_x + S_y) % 2 == (T_x + T_y) % 2:
        return max(abs(T_x - S_x), abs(T_y - S_y))
    else:
        return abs(T_x - S_x) + abs(T_y - S_y)

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    S_x, S_y, T_x, T_y = map(int, data)
    result = minimum_toll(S_x, S_y, T_x, T_y)
    print(result)