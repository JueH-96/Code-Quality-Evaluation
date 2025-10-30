class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        online = [True] * numberOfUsers
        offline_times = {}

        for event in events:
            event_type = event[0]
            timestamp = int(event[1])

            # Update online status based on offline events
            for user_id in range(numberOfUsers):
                if user_id in offline_times and timestamp >= offline_times[user_id]:
                    online[user_id] = True
                    del offline_times[user_id]

            if event_type == "OFFLINE":
                user_id = int(event[2])
                online[user_id] = False
                offline_times[user_id] = timestamp + 60

            elif event_type == "MESSAGE":
                mentions_string = event[2]
                mentioned_users = set()

                if mentions_string == "ALL":
                    for user_id in range(numberOfUsers):
                        mentions[user_id] += 1
                elif mentions_string == "HERE":
                    for user_id in range(numberOfUsers):
                        if online[user_id]:
                            mentions[user_id] += 1
                else:
                    ids = mentions_string.split()
                    for id_str in ids:
                        user_id = int(id_str[2:])
                        mentions[user_id] += 1

        return mentions