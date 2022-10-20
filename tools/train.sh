type=$1
shift
override=$@
gpus="2,3"

if [[ $type == "whole" ]]; then

    config=$type
    for f in {0..3}; do
        python ./Solver.py --config ./configs/${config}.yaml --gpus $gpus --override fold=$f $override
    done

elif [[ $type == hos* ]]; then

    config=$type
    if [[ ! -f ./configs/${config}.yaml ]]; then
        echo "unsupported"
        exit;
    fi
    for hos in {0..3}; do
        if [[ $type == *smp ]]; then
            if [[ hos -lt 2 ]]; then 
                N=4
            elif [[ hos -lt 3 ]]; then
                N=2
            else
                N=1
            fi
            N="data.sampler.multiple_times=$N"
        else
            N=""
        fi
        for f in {0..3}; do
            python ./Solver.py --config ./configs/${config}.yaml --gpus $gpus --override fold=$f hos_id=$hos $N $override
        done
    done
    

elif [[ $type == fed* ]]; then 

    config=$type
    if [[ ! -f ./configs/${config}/_base_.yaml ]]; then
        echo "unsupported"
        exit;
    fi
    for f in {0..3}; do
        python FedSolver.py --config-dir ./configs/${config} >/dev/null 2>&1 &
        running="$!"
        sleep 2
        for hos in {0..3}; do
            if [[ hos -eq 0 ]]; then
                python ./Solver.py --config ./configs/${config}/${hos}.yaml --gpus $gpus --override fold=$f $override &
            else
                python ./Solver.py --config ./configs/${config}/${hos}.yaml --gpus $gpus --override fold=$f $override >/dev/null 2>&1 &
            fi
            running="${running} $!"
        done
        wait $running
    done

else
    echo "unsupported"
fi