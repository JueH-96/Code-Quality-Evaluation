from collections import defaultdict

def make_string_good(s: str) -> int:
    counts = defaultdict(int)
    for c in s:
        counts[c] += 1

    min_steps = float('inf')

    max_t = 20000 + 26  # A reasonable upper limit for t

    for t in range(0, max_t + 1):
        # Compute initial sum of absolute differences
        initial_sum = sum(abs(v - t) for v in counts.values())

        # Initialize transfer steps
        transfer = 0
        current_counts = counts.copy()

        # Process each character in forward order
        for c in 'abcdefghijklmnopqrstuvwxyz':
            current_count = current_counts[c]

            if current_count > t:
                excess = current_count - t
                next_char = chr(ord(c) + 1)
                if next_char <= 'z':
                    deficit = current_counts.get(next_char, 0)
                    transfer += min(excess, deficit)
                    current_counts[c] -= transfer
                    current_counts[next_char] += transfer

        # Process each character in reverse order
        for c in reversed('abcdefghijklmnopqrstuvwxyz'):
            current_count = current_counts[c]

            if current_count < t:
                deficit = t - current_count
                prev_char = chr(ord(c) - 1)
                if prev_char >= 'a':
                    excess = current_counts.get(prev_char, 0)
                    transfer += min(deficit, excess)
                    current_counts[c] -= transfer
                    current_counts[prev_char] += transfer

        # Calculate steps
        steps = initial_sum - transfer
        if steps >= 0 and steps < min_steps:
            min_steps = steps

    return min_steps