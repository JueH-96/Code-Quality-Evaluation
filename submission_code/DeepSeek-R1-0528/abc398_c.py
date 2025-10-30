from collections import Counter

def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    count_map = Counter(arr)
    
    candidate_index = -1
    candidate_value = -1
    
    for i, num in enumerate(arr):
        if count_map[num] == 1:
            if num > candidate_value:
                candidate_value = num
                candidate_index = i + 1
                
    print(candidate_index)

if __name__ == "__main__":
    main()