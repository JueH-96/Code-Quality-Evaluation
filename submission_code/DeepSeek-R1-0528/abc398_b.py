from collections import Counter

def main():
    arr = list(map(int, input().split()))
    counts = Counter(arr)
    at_least_two = sum(1 for cnt in counts.values() if cnt >= 2)
    triplet_exists = any(cnt >= 3 for cnt in counts.values())
    
    if at_least_two >= 2 and triplet_exists:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()