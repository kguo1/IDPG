TOTAL_NUM_UPDATES=7780  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=420      # 6 percent of the number of updates
LR=1e-03                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
       # Batch size.

ROBERTA_PATH=$1
SAVE=$2
seed=$3
ARCH=roberta_base
pdim=$4
node=$5
prefixlen=$6
insertposition=$7
LR=$8
gq=16
custom_insert_position_fraction=${10}
distributed_world_size=${11}
MAX_SENTENCES=${12}
MAX_TOKENS=${13}
ENCODER_EMBED_DIM=${14}
FFN_DIM=${15}
layers=${16}


## we need to update distributed port set to 1024 up to 65535


echo $ROBERTA_PATH
echo $SAVE
echo $seed
echo $ARCH
echo $node "cuda visible devices"
echo $custom_insert_position_fraction "custom insert position fraction"


echo $ENCODER_EMBED_DIM "encoder embed dim"

echo $FFN_DIM "ffn dimension"

echo $layers "ffn layers"


mkdir -p ${SAVE}
#--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
#    --freeze-encoder \

## --memory-efficient-bf16 \
## --memory-efficient-fp16 \
##--model-parallel-size $distributed_world_size\

CUDA_VISIBLE_DEVICES=$node fairseq-train RTE-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens $MAX_TOKENS \
    --task sentence_prediction \
    --memory-efficient-fp16 \
    --encoder-embed-dim $ENCODER_EMBED_DIM \
    --encoder-ffn-embed-dim $FFN_DIM \
    --encoder-layers $layers \
    --empty-cache-freq 100 \
    --update-freq 4 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch ${ARCH} \
    --criterion sentence_prediction \
    --freeze-encoder \
    --distributed-world-size $distributed_world_size \
    --custom_insert_position_fraction $custom_insert_position_fraction \
    --add-suffix --suffix-len $prefixlen --prompt-generation --generation-freeze --insert-position $insertposition --generation-layer 2 --generation-quaternions $gq --middle-prompt-insert-layer 1 --middle-previous --middle-prompt-mode layerb --phm-bottleneck-dim $pdim \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --seed ${seed}\
    --lr-scheduler fixed --lr $LR --max-update $TOTAL_NUM_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 50 \
    --find-unused-parameters \
    --save-dir ${SAVE} \
    --no-epoch-checkpoints --no-last-checkpoints \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
