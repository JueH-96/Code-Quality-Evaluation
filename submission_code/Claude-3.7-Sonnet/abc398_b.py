from collections import Counter

def main():
    cards = list(map(int, input().split()))
    
    # Count occurrences of each number
    counts = Counter(cards)
    
    # Check for full house
    for num1, count1 in counts.items():
        if count1 >= 3:
            for num2, count2 in counts.items():
                if num2 != num1 and count2 >= 2:
                    print("Yes")
                    return
    
    print("No")

if __name__ == "__main__":
    main()