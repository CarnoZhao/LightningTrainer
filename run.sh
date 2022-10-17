type=$1

if [ $type == "whole" ]; then

    config=whole
    for f in {0..3}; do
        sed -i -E -e "s/((fold|version|load_from).*)[0-4]([^0-9]*)$/\1$f\3/g" ./configs/${config}.yaml
        python ./Solver.py --config ./configs/${config}.yaml --gpus 2,3
    done

elif [ $type == "hos" ]; then

    config=hos
    for hos in {0..3}; do
        sed -i -E -e "/^name/! s/((hos)[^0-9]*)[0-4](.*)$/\1$hos\3/g" ./configs/${config}.yaml
        for f in {0..3}; do
            sed -i -E -e "/^name/! s/((fold|version|load_from).*)[0-4]([^0-9]*)$/\1$f\3/g" ./configs/${config}.yaml
            python ./Solver.py --config ./configs/${config}.yaml --gpus 2,3
        done
    done

elif [ $type == "fed" ]; then 

    config=fed
    for f in {0..3}; do
        sed -i -E -e "s/((fold|version|load_from).*)[0-4]([^0-9]*)$/\1$f\3/g" ./configs/${config}/*.yaml
        python FedSolver.py --config-dir ./configs/${config} >/dev/null 2>&1 &
        running="$!"
        sleep 2
        for hos in {0..3}; do
            if [[ hos -eq 0 ]]; then
                python ./Solver.py --config ./configs/${config}/${hos}.yaml --gpus 2,3 &
            else
                python ./Solver.py --config ./configs/${config}/${hos}.yaml --gpus 2,3 >/dev/null 2>&1 &
            fi
            running="${running} $!"
        done
        wait $running
    done

else
    echo "unsupported"
fi