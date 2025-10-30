# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    beans = []
    for i in range(N):
        A = int(data[2 * i + 1])
        C = int(data[2 * i + 2])
        beans.append((A, C))
    
    color_to_deliciousness = {}
    
    for deliciousness, color in beans:
        if color not in color_to_deliciousness:
            color_to_deliciousness[color] = []
        color_to_deliciousness[color].append(deliciousness)
    
    max_min_deliciousness = 0
    
    for deliciousness_list in color_to_deliciousness.values():
        min_deliciousness = min(deliciousness_list)
        if min_deliciousness > max_min_deliciousness:
            max_min_deliciousness = min_deliciousness
    
    print(max_min_deliciousness)

if __name__ == "__main__":
    main()