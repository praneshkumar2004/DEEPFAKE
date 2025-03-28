import qrcode

def generate_qr():
    qr = qrcode.qrcode(
        version=3,
        box_size=20,
        border=10,
        error_correction=qrcode.constants.ERROR_CORRECT_H
    )
    qr.add_data("https://t.me/DeepFakeIntelli_bot")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("telegram_bot_qr.png")
    print("QR code generated successfully as 'telegram_bot_qr.png'")

if __name__ == "__main__":
    generate_qr()