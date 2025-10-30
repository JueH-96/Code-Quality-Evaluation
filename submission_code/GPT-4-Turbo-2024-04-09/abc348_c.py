import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    beans = data[1:]
    
    from collections import defaultdict
    color_to_deliciousness = defaultdict(list)
    
    for i in range(N):
        A = int(beans[2 * i])
        C = int(beans[2 * i + 1])
        color_to_deliciousness[C].append(A)
    
    max_min_deliciousness = 0
    for deliciousness_list in color_to_deliciousness.values():
        min_deliciousness = min(deliciousness_list)
        if min_deliciousness > max_min_deliciousness:
            max_min_deliciousness = min_deliciousness
    
    print(max_min_deliciousness)

if __name__ == "__main__":
    main()