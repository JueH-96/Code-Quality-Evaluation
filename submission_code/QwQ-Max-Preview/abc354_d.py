def count_mod4(a, b, c):
    if a > b:
        return 0
    first = a + ((c - a % 4) + 4) % 4
    if first > b:
        return 0
    last = b - ((b - c) % 4)
    return ((last - first) // 4) + 1

def main():
    A, B, C, D = map(int, input().split())
    m_start = B
    m_end = D - 1
    if m_start > m_end:
        even_count = 0
        odd_count = 0
    else:
        even_count = (m_end // 2 - m_start // 2) + (1 if m_start % 2 == 0 else 0)
        total_m = m_end - m_start + 1
        odd_count = total_m - even_count

    group0_count = count_mod4(A, C-1, 0)
    group2_count = count_mod4(A, C-1, 1)
    group3_count = count_mod4(A, C-1, 3)

    total = group0_count * (D - B) + group2_count * odd_count + group3_count * even_count
    print(2 * total)

if __name__ == "__main__":
    main()