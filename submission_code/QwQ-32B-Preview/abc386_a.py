def can_form_full_house(A, B, C, D):
    from collections import defaultdict
    
    freq = defaultdict(int)
    freq[A] += 1
    freq[B] += 1
    freq[C] += 1
    freq[D] += 1
    
    for x in range(1, 14):
        freq[x] += 1  # Add one card of value x
        counts = list(freq.values())
        count_triples = counts.count(3)
        count_pairs = counts.count(2)
        if count_triples == 1 and count_pairs == 1:
            return True
        freq[x] -= 1  # Revert the change
    return False

def main():
    input_line = input().strip()
    parts = input_line.split()
    if len(parts) != 4:
        raise ValueError("Exactly four integers are required.")
    A, B, C, D = map(int, parts)
    if not all(1 <= x <= 13 for x in [A, B, C, D]):
        raise ValueError("All input values must be integers between 1 and 13.")
    
    if can_form_full_house(A, B, C, D):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()