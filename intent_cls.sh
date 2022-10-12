test_file="./data/intent/test.json"
ckpt_path="./ckpt/intent/intent_model.pt"

echo "Start testing..."
python test_intent.py --test_file ${test_file} --ckpt_path ${ckpt_path}
echo "Finish testing..."



