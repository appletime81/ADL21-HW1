# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# python3 test_slot.py --test_file ...

test_file="./data/slot/test.json"

echo "Start testing..."
python test_slot.py --test_file ${test_file} --device cuda --batch_size 1 --max_len 40 --num_layers 4
echo "Finish testing..."
