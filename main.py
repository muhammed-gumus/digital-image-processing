import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():
    print("Sayısal Görüntü İşleme Projesine Hoş Geldiniz!")

    while True:
        image_filename = input("Lütfen resim dosyasının adını girin: ")
        image = np.array(Image.open(image_filename))
        Image.fromarray(image).save("input_image.jpg")
        print("Orijinal resim başarıyla kaydedildi.")

        skip_second_part = False
        while not skip_second_part:
            print("\nÖn İşlem Seçenekleri:")
            print("(a) Renkli resmi Gri seviye resme dönüştürme")
            print("(b) Gri resmi Siyah Beyaz resme dönüştürme (Eşik girerek)")
            print("(c) Zoom in – Zoom out")
            print("(d) Resimden istenilen bölgenin kesilip alınması")
            print("(e) Ön işlem uygulamak istemiyorum")

            option = input("Lütfen bir seçenek girin (a/b/c/d/e): ")

            if option == 'a':
                image = convert_to_grayscale(image)
            elif option == 'b':
                threshold = int(input("Lütfen bir eşik değeri girin: "))
                image = convert_to_binary(image, threshold)
            elif option == 'c':
                image = zoom_in(image, 2.0)
            elif option == 'd':
                image = crop_region(image)
            elif option == 'e':
                skip_second_part = True
            else:
                print("Geçersiz seçenek! Lütfen tekrar deneyin.")

            if not skip_second_part:
                while True:
                    print("\nÖn İşlem Uygulama Seçenekleri:")
                    print("(a) Histogram Oluşturma")
                    print(
                        "(b) Histogram Eşitleme (Equalization - Gri seviyeye resimde)")
                    print("(c) Görüntü Nicemleme (Quantization - Ton Sayısı Seçilerek)")
                    print("(d) Ön işlem uygulamak istemiyorum")

                    preprocessing_option = input(
                        "Lütfen bir seçenek girin (a/b/c/d): ")

                    if preprocessing_option == 'a':
                        image = create_histogram(image)
                    elif preprocessing_option == 'b':
                        image = equalize_histogram(image)
                    elif preprocessing_option == 'c':
                        image = quantization(image)
                    elif preprocessing_option == 'd':
                        break
                    else:
                        print("Geçersiz seçenek! Lütfen tekrar deneyin.")

                print("İlerleyen kısımlar için devam ediyoruz...")


def save_image(image, filename):
    Image.fromarray(image).save(filename)
    print(f"{filename} başarıyla kaydedildi.")


def convert_to_grayscale(image):
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    save_image(gray_image.astype(np.uint8), "grayscale_image.jpg")
    print("Renkli resim gri seviye resme dönüştürüldü.")
    return gray_image


def convert_to_binary(image, threshold):
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    binary_image = (gray_image > threshold) * 255
    save_image(binary_image.astype(np.uint8), "binary_image.jpg")
    print("Gri resim siyah beyaz resme dönüştürüldü.")
    return binary_image


def zoom_in(img, zoom, coord=None):
    h, w, _ = [int(zoom * i) for i in img.shape]

    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [int(zoom * c) for c in coord]

    img = np.array(Image.fromarray(img).resize((w, h)))
    img = img[int(round(cy - h / (2 * zoom))):int(round(cy + h / (2 * zoom))),
              int(round(cx - w / (2 * zoom))):int(round(cx + w / (2 * zoom))), :]
    save_image(img, "zoom_image.jpg")
    return img


def crop_region(image):
    start_x = int(input("Başlangıç x koordinatını girin: "))
    start_y = int(input("Başlangıç y koordinatını girin: "))
    end_x = int(input("Bitiş x koordinatını girin: "))
    end_y = int(input("Bitiş y koordinatını girin: "))
    cropped_image = image[start_y:end_y, start_x:end_x]
    save_image(cropped_image, "cropped_image.jpg")
    print("Belirtilen bölge başarıyla kesildi.")
    return cropped_image


def create_histogram(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Piksel Sayısı')
    plt.show()
    save_image(image, "histogram_image.jpg")
    return image


def equalize_histogram(image):
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    cdf = np.cumsum(np.histogram(gray_image, bins=256, range=[0, 256])[0])
    cdf_normalized = cdf / cdf[-1]
    equalized_image = np.interp(gray_image, range(
        256), cdf_normalized * 255).astype(np.uint8)
    save_image(equalized_image, "equalized_image.jpg")
    plt.subplot(121), plt.imshow(image), plt.title('Orijinal')
    plt.subplot(122), plt.imshow(equalized_image,
                                 cmap='gray'), plt.title('Histogram Eşitleme')
    plt.show()
    return equalized_image


def quantization(image):
    quantized_image = (image // 50) * 50
    save_image(quantized_image.astype(np.uint8), "quantized_image.jpg")
    print("Görüntü nicemleme tamamlandı.")
    return quantized_image


if __name__ == "__main__":
    main()
