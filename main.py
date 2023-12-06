import cv2
import matplotlib.pyplot as plt


def main():
    print("Sayısal Görüntü İşleme Projesine Hoş Geldiniz!")

    while True:
        # Kullanıcıdan resim dosyasının adını alın
        image_filename = input("Lütfen resim dosyasının adını girin: ")

        # Resimi oku ve program dizinine kaydet
        image = cv2.imread(image_filename)
        cv2.imwrite("input_image.jpg", image)

        print("Resim başarıyla kaydedildi.")

        # Ön işlem seçenekleri
        skip_second_part = False  # İkinci kısmı atlamak için bayrak
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
                image = zoom_in_out(image, 2.0)
            elif option == 'd':
                image = crop_region(image)
            elif option == 'e':
                skip_second_part = True
            else:
                print("Geçersiz seçenek! Lütfen tekrar deneyin.")

            # Ön işlem uygulanacaksa
            if not skip_second_part:
                while True:
                    print("\nÖn İşlem Uygulama Seçenekleri:")
                    print("(a) Histogram Oluşturma")
                    print("(b) Histogram Eşitleme (Equalization - Gri seviyeye resimde)")
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

                # İlerleyen kısımları buraya ekleyebilirsiniz.
                print("İlerleyen kısımlar için devam ediyoruz...")


def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale_image.jpg", gray_image)
    print("Renkli resim gri seviye resme dönüştürüldü.")
    return gray_image


def convert_to_binary(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binary_image.jpg", binary_image)
    print("Gri resim siyah beyaz resme dönüştürüldü.")
    return binary_image


def zoom_in_out(img, zoom, coord=None):
    # Translate to zoomed coordinates
    h, w, _ = [ zoom * i for i in img.shape ]
    
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]
    
    img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
               int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
               : ]
    cv2.imwrite("zoom_image.jpg", img)
    return img


def crop_region(image):
    # Kullanıcıdan başlangıç ve bitiş koordinatlarını al
    start_x = int(input("Başlangıç x koordinatını girin: "))
    start_y = int(input("Başlangıç y koordinatını girin: "))
    end_x = int(input("Bitiş x koordinatını girin: "))
    end_y = int(input("Bitiş y koordinatını girin: "))

    # Belirtilen bölgeyi kesip al
    cropped_image = image[start_y:end_y, start_x:end_x]
    cv2.imwrite("cropped_image.jpg", cropped_image)
    print("Belirtilen bölge başarıyla kesildi.")
    return cropped_image


def create_histogram(image):
    # Histogram oluşturma işlemi buraya eklenecek
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Piksel Sayısı')
    plt.show()
    return image


def equalize_histogram(image):
    # Resmi gri seviyeye dönüştür
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram eşitleme işlemi
    equalized_image = cv2.equalizeHist(gray_image)

    # Görüntüyü renkliye çevir (Matplotlib için)
    equalized_image_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

    # Histogram grafiğini çiz
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Orijinal')
    plt.subplot(122), plt.imshow(cv2.cvtColor(equalized_image_bgr, cv2.COLOR_BGR2RGB)), plt.title('Histogram Eşitleme')
    plt.show()

    return equalized_image


def quantization(image):
    # Görüntü nicemleme işlemi buraya eklenecek
    # Örneğin, piksel değerlerini belirli bir ton sayısına indirgeme
    quantized_image = image // 50 * 50  # 50'şer piksel değerine nicemleme örneği
    cv2.imwrite("quantized_image.jpg", quantized_image)
    print("Görüntü nicemleme tamamlandı.")
    return quantized_image


if __name__ == "__main__":
    main()
