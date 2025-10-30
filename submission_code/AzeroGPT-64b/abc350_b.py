# import sys

# if __name__ == '__main__':
#     lines = []
#     for line in sys.stdin:
#         lines.append(line)
#     n, q = map(int, lines[0].split())
#     t = list(map(int, lines[1].split()))
#     teeth = [1] * n
#     for i in t:
#         if teeth[i-1]:
#             teeth[i-1] = 0
#         else:
#             teeth[i-1] = 1
#     print(teeth.count(1))
n, q = map(int, input().split())
t = list(map(int, input().split()))
teeth = [1] * n
for i in t:
    if teeth[i-1]:
        teeth[i-1] = 0
    else:
        teeth[i-1] = 1
print(teeth.count(1))