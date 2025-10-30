def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    initial_votes_sum = sum(a)
    remaining_votes = k - initial_votes_sum
    results = []
    for i_candidate_index in range(n):
        min_votes_needed = -1
        found_victory = False
        if is_guaranteed_victory(n, m, k, list(a), i_candidate_index, 0, remaining_votes):
            min_votes_needed = 0
            found_victory = True
        else:
            low = 1
            high = remaining_votes
            possible_votes = -1
            while low <= high:
                mid_votes = (low + high) // 2
                if is_guaranteed_victory(n, m, k, list(a), i_candidate_index, mid_votes, remaining_votes):
                    possible_votes = mid_votes
                    high = mid_votes - 1
                    found_victory = True
                else:
                    low = mid_votes + 1
            if found_victory:
                min_votes_needed = possible_votes
                
        results.append(min_votes_needed)
        
    print(*(results))

def is_guaranteed_victory(n, m, k, current_votes, candidate_index, additional_votes, total_remaining_votes):
    candidate_votes = current_votes[candidate_index] + additional_votes
    votes_to_distribute = total_remaining_votes - additional_votes
    other_candidates_indices = [i for i in range(n) if i != candidate_index]
    votes_needed = []
    for j_index in other_candidates_indices:
        needed = max(0, candidate_votes + 1 - current_votes[j_index])
        if needed > 0:
            votes_needed.append(needed)
    
    votes_needed.sort()
    votes_spent = 0
    candidates_exceeding = 0
    for needed_votes in votes_needed:
        if votes_spent + needed_votes <= votes_to_distribute:
            votes_spent += needed_votes
            candidates_exceeding += 1
        else:
            break
            
    return candidates_exceeding < m

if __name__ == '__main__':
    solve()