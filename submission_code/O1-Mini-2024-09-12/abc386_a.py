from collections import Counter

def can_form_full_house(cards):
    counts = Counter(cards)
    if len(counts) == 2:
        cnt = list(counts.values())
        if (3 in cnt and 1 in cnt) or (2 in cnt and 2 in cnt):
            return "Yes"
    return "No"

def main():
    cards = list(map(int, input().split()))
    print(can_form_full_house(cards))

if __name__ == "__main__":
    main()