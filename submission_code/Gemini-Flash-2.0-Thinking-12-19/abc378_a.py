import sys

def solve():
    colors_str = sys.stdin.readline().strip().split()
    colors = [int(c) for c in colors_str]
    color_counts = [0] * 5
    for color in colors:
        color_counts[color] += 1
    
    operations = 0
    for i in range(1, 5):
        operations += color_counts[i] // 2
        
    print(operations)

if __name__ == '__main__':
    solve()