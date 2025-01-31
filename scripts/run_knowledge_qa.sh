categories=("sports", "art", "technology", "celebrity")

for category in ${categories[@]}; do
    python datasets/create_knowledge_qa.py --category $category --num_pairs 200
done