# # g++ -std=c++11 -shared zero_out_test.cc -o zero_out_test.so -I ${TF_INC} -fPIC
#
# zero_out_module = tf.load_op_library('/home/firiuza/PycharmProjects/TensorflowPractice1/Lesson13/zero_out.so')
# zero_out = zero_out_module.zero_out

# (tf2) firiuza@firiuza-System-Product-Name:~/anaconda2/envs/tf2$ g++ -std=c++11 -shared /home/firiuza/PycharmProjects/TensorflowPractice1/Lesson13/zero_out.cc -o/home/firiuza/PycharmProjects/TensorflowPractice1/Lesson13/zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('/home/firiuza/PycharmProjects/TensorflowPractice1/Lesson13/zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.numpy(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()