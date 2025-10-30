A, B = map(int, input().split())

x_list = [A + (B - A), A - (B - A), B + (B - A)]

print(len(set(x_list)))