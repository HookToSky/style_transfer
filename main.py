import datetime
import tensorflow as tf

from tensorboard.plugins.hparams import api as hp
from absl import app

from src.models.loss import style_content_loss
from src.models.preprocess import clip_0_1
from src.models.model import Style2ContentModel
from src.visualization.visualize import img_to_tensor, tensor_to_img

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'summarylogs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
tf.debugging.experimental.enable_dump_debug_info(
    train_log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
    )


STYLE_WEIGHT=hp.HParam('style_weight', hp.Discrete([1e-3, 1e-2, 1e-1]))
CONTENT_WEIGHT=hp.HParam('content_weight', hp.Discrete([1e2, 1e3, 1e4]))
# LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([0.001, 0.05, 0.01]))
TOTAL_VAR_WEIGHT=hp.HParam('total_variation_weight', hp.Discrete([20, 30, 35]))

HPARAMS = [
    STYLE_WEIGHT,
    CONTENT_WEIGHT,
    #LEARNING_RATE,
    TOTAL_VAR_WEIGHT
]

content_layers = ['block5_conv2']
style_layers =['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

@tf.function()
def train_step(image, style_targets, content_targets, opt, sw, cw, tvw):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, sw, style_layers, style_targets, cw, content_layers, content_targets)
        loss += tvw*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

    train_loss(loss)

content_tf_img = img_to_tensor('./data/interim/content/landscape2.jpeg')
style_tf_img = img_to_tensor('./data/interim/style/monet1.jpeg')

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
extractor = Style2ContentModel(style_layers, content_layers)

def run(content_targets, style_targets, params, opt):
    img = tf.Variable(content_tf_img)
    step = 0

    for n in range(12):
        with train_summary_writer.as_default():
            hp.hparams(params)
            tf.summary.scalar('loss', train_loss.result(), step=n)
            tf.summary.image("Combined Image", img, step=n)
            tf.summary.trace_on(
                graph=True, profiler=False
            )
            tf.summary.trace_export(
                name="nst_trace",
                step=1,
                profiler_outdir=train_log_dir)
            for _ in range(120):
                step += 1
                train_step(img, style_targets, content_targets, opt, params[STYLE_WEIGHT], params[CONTENT_WEIGHT], params[TOTAL_VAR_WEIGHT])
        print("Train step: {}".format(step))
        template = 'Epoch {}, Loss: {}'
        print (template.format(n+1, train_loss.result()))
        train_loss.reset_states()


def main(content_img, style_img):
    session_num = 0
    style_targets = extractor(style_img)['style']
    content_targets = extractor(content_img)['content']
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.95, epsilon=1e-5)
    for cw in CONTENT_WEIGHT.domain.values:
        for sw in STYLE_WEIGHT.domain.values:
            for tvw in TOTAL_VAR_WEIGHT.domain.values:
                # for lr in LEARNING_RATE.domain.values:
                    
                hparam = {
                    STYLE_WEIGHT: sw,
                    CONTENT_WEIGHT: cw,
                    #LEARNING_RATE: lr,
                    TOTAL_VAR_WEIGHT: tvw
                    
                }
                
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparam[h] for h in hparam})
                run(content_targets, style_targets, hparam, opt)
                session_num += 1
   

if __name__ == "__main__":
    app.run(main(content_tf_img, style_tf_img))