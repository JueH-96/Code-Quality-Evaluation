def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    remaining_votes = k - sum(a)
    if remaining_votes < 0:
        remaining_votes = 0
    
    results = []
    for i_candidate_index in range(n):
        candidate_id = i_candidate_index + 1
        min_votes_needed = -1
        low = 0
        high = remaining_votes
        
        while low <= high:
            x = (low + high) // 2
            candidate_votes = a[i_candidate_index] + x
            other_candidates_votes_needed = []
            for j_candidate_index in range(n):
                if i_candidate_index == j_candidate_index:
                    continue
                needed_votes = max(0, candidate_votes - a[j_candidate_index] + 1)
                other_candidates_votes_needed.append((needed_votes, j_candidate_index))
            
            other_candidates_votes_needed.sort(key=lambda pair: pair[0])
            
            votes_spent = 0
            count_surpassing = 0
            for needed_vote, index in other_candidates_votes_needed:
                if votes_spent + needed_vote <= remaining_votes - x:
                    votes_spent += needed_vote
                    count_surpassing += 1
                else:
                    break
                    
            if count_surpassing < m:
                min_votes_needed = x
                high = x - 1
            else:
                low = x + 1
                
        results.append(min_votes_needed)
        
    print(*(results))

if __name__ == '__main__':
    solve()