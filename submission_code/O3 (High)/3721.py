from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        """
        Processes the list of events and returns how many times each user is
        mentioned.
        """
        # ------------------------------------------------------------------ #
        # 1.  Sort events chronologically; for equal timestamps make sure
        #     OFFLINE events are handled before MESSAGE events, exactly as
        #     required by the statement.
        # ------------------------------------------------------------------ #
        sorted_events = sorted(
            enumerate(events),
            key=lambda x: (int(x[1][1]), 0 if x[1][0] == "OFFLINE" else 1, x[0])
        )
        
        N = numberOfUsers
        # offline_until[u] == time when user u becomes online again
        offline_until = [0] * N
        
        # mentions[u] == total number of times user u was mentioned
        mentions = [0] * N
        
        # ------------------------------------------------------------------ #
        # 2.  Process every event in the sorted order.
        # ------------------------------------------------------------------ #
        for _, event in sorted_events:
            etype, ts_str, payload = event
            t = int(ts_str)
            
            if etype == "OFFLINE":
                # user becomes offline for 60 time-units, starting now (inclusive)
                uid = int(payload)
                offline_until[uid] = t + 60
                continue
            
            # ---------------- MESSAGE event --------------------------------
            tokens = payload.split()
            for token in tokens:
                if token == "ALL":
                    # All users, even the offline ones
                    for u in range(N):
                        mentions[u] += 1
                        
                elif token == "HERE":
                    # Only users currently online
                    for u in range(N):
                        if offline_until[u] <= t:
                            mentions[u] += 1
                            
                elif token.startswith("id"):
                    uid = int(token[2:])          # token == 'id<number>'
                    mentions[uid] += 1
                # any other token is impossible according to the constraints

        return mentions