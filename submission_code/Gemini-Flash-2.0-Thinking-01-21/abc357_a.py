def solve():
    n, m = map(int, input().split())
    h_list = list(map(int, input().split()))
    remaining_disinfectant = m
    successful_aliens_count = 0
    for hands_needed in h_list:
        if remaining_disinfectant >= hands_needed:
            remaining_disinfectant -= hands_needed
            successful_aliens_count += 1
        else:
            remaining_disinfectant = 0
    print(successful_aliens_count)

if __name__ == '__main__':
    solve()